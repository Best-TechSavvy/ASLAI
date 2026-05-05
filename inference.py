# ============================================================
# inference.py
# Real-time ASL inference pipeline
# Connects webcam landmark extractor → model → gloss output
# ============================================================

import json
import time
import collections
import numpy as np
import torch
from pathlib import Path
from model import load_model


# ============================================================
# Normalization (matches training preprocessing exactly)
# ============================================================
NUM_FACE, NUM_POSE, NUM_HAND = 76, 12, 21

def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """
    Normalize a (T, 130, 3) landmark sequence.
    Matches the normalization used during training.
    """
    seq = seq.copy().astype(np.float32)
    LHAND = NUM_FACE + NUM_POSE      # 88
    RHAND = LHAND + NUM_HAND         # 109
    PS    = NUM_FACE                 # 76

    # Hands: center on wrist, scale by hand span
    for hs in [LHAND, RHAND]:
        h  = seq[:, hs:hs+NUM_HAND, :]
        w  = h[:, 0:1, :]
        hc = h - w
        sc = np.nanmax(np.linalg.norm(hc, axis=-1, keepdims=True),
                       axis=1, keepdims=True) + 1e-6
        seq[:, hs:hs+NUM_HAND, :] = hc / sc

    # Pose: center on mid-shoulder, scale by shoulder width
    p  = seq[:, PS:PS+NUM_POSE, :]
    ls, rs = p[:, 0:1, :], p[:, 1:2, :]
    mid = (ls + rs) / 2.0
    sw  = np.linalg.norm(ls - rs, axis=-1, keepdims=True) + 1e-6
    seq[:, PS:PS+NUM_POSE, :] = (p - mid) / sw

    # Face: center on nose tip, scale by inter-eye distance
    f    = seq[:, 0:NUM_FACE, :]
    nose = f[:, 40:41, :]
    lec  = f[:, 42:50, :].mean(axis=1, keepdims=True)
    rec  = f[:, 50:58, :].mean(axis=1, keepdims=True)
    ed   = np.linalg.norm(lec - rec, axis=-1, keepdims=True) + 1e-6
    seq[:, 0:NUM_FACE, :] = (f - nose) / ed

    # Replace NaN and any infinities produced by near-zero scale factors
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# CTC greedy decoder
# ============================================================
def greedy_ctc_decode(log_probs: np.ndarray,
                      idx_to_char: dict,
                      blank_idx: int = 0) -> str:
    """
    Greedy CTC decoder — collapses repeated tokens and removes blanks.

    Args:
        log_probs  : (T, vocab_size) numpy array
        idx_to_char: mapping from index → character
        blank_idx  : index of the blank token (default 0)

    Returns:
        decoded string
    """
    tokens = log_probs.argmax(axis=-1).tolist()
    decoded, prev = [], None
    for t in tokens:
        if t != blank_idx and t != prev:
            decoded.append(idx_to_char.get(t, '?'))
        prev = t
    return ''.join(decoded)


# ============================================================
# ASL Inference Engine
# ============================================================
class ASLInferenceEngine:
    """
    Real-time ASL inference engine.

    Usage:
        engine = ASLInferenceEngine('weights/asl_model_weights_v2.pt',
                                    'weights/vocabulary.json',
                                    'weights/sign_to_idx.json')

        # Feed one frame at a time from your webcam extractor
        engine.add_frame(landmarks_130x3)

        # Get current predictions
        result = engine.predict()
        print(result['fingerspelling'])   # e.g. "HELLO"
        print(result['top_sign'])         # e.g. "HAPPY"
    """

    def __init__(self,
                 weights_path: str,
                 vocab_path:   str,
                 signs_path:   str,
                 device:       str = 'cpu',
                 buffer_size:  int = 64,
                 min_frames:   int = 10):
        """
        Args:
            weights_path : path to asl_model_weights_v2.pt
            vocab_path   : path to vocabulary.json
            signs_path   : path to sign_to_idx.json
            device       : 'cpu' or 'cuda'
            buffer_size  : number of recent frames to keep (sliding window)
            min_frames   : minimum frames before running inference
        """
        self.device      = device
        self.buffer_size = buffer_size
        self.min_frames  = min_frames

        # Load model
        print(f'Loading model from {weights_path}...')
        self.model = load_model(weights_path, device)
        print(f'Model loaded ✅  (device: {device})')

        # Load vocabularies
        with open(vocab_path) as f:
            vocab = json.load(f)
        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
        self.blank_idx   = vocab.get('blank_idx', 0)

        with open(signs_path) as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}

        # Sliding window buffer of (130, 3) frames
        self.frame_buffer = collections.deque(maxlen=buffer_size)

        # Cache last result to avoid redundant inference
        self._last_result   = None
        self._last_n_frames = 0

        # Track how many frames have ever been added
        self._frame_counter = 0

        # Dedicated flag to force a cache bypass on the next predict() call.
        # Used exclusively by get_timing() — avoids corrupting _last_n_frames
        # with a sentinel value.
        self._force_rerun = False

        print(f'Vocabulary  : {len(self.char_to_idx)} chars + blank')
        print(f'Signs       : {len(self.idx_to_sign)} classes')
        print(f'Buffer size : {buffer_size} frames')

    def add_frame(self, landmarks: np.ndarray):
        """
        Add one frame of landmarks to the buffer.

        Args:
            landmarks : (130, 3) numpy array from extract_landmarks.py
        """
        if landmarks.shape != (130, 3):
            raise ValueError(f'Expected landmarks shape (130, 3), got {landmarks.shape}')
        self.frame_buffer.append(landmarks.copy())
        self._frame_counter += 1

    def add_sequence(self, landmarks: np.ndarray):
        """
        Add multiple frames at once (e.g. from a video file).

        Args:
            landmarks : (T, 130, 3) numpy array
        """
        if landmarks.ndim != 3 or landmarks.shape[1:] != (130, 3):
            raise ValueError(
                f'Expected landmarks shape (T, 130, 3), got {landmarks.shape}'
            )
        for frame in landmarks:
            self.frame_buffer.append(frame.copy())
            self._frame_counter += 1

    def clear(self):
        """Clear the frame buffer and cached results."""
        self.frame_buffer.clear()
        self._last_result   = None
        self._last_n_frames = 0
        self._frame_counter = 0
        self._force_rerun   = False

    @torch.no_grad()
    def predict(self) -> dict:
        """
        Run inference on the current frame buffer.

        Returns dict with keys:
            fingerspelling : str   — decoded character sequence (uppercase)
            top_sign       : str   — most likely ASL sign (uppercase)
            top_sign_conf  : float — confidence 0-1
            top5_signs     : list of (sign_name, confidence) tuples
            n_frames       : int   — number of frames used
            ready          : bool  — False if not enough frames yet
        """
        n = len(self.frame_buffer)

        if n < self.min_frames:
            return {
                'fingerspelling': '',
                'top_sign'      : '',
                'top_sign_conf' : 0.0,
                'top5_signs'    : [],
                'n_frames'      : n,
                'ready'         : False,
            }

        # Return cached result if no new frames and no forced rerun
        if (not self._force_rerun
                and self._frame_counter == self._last_n_frames
                and self._last_result is not None):
            return self._last_result

        self._force_rerun = False   # consume the flag

        # Build sequence array from buffer
        seq = np.stack(list(self.frame_buffer), axis=0)  # (T, 130, 3)
        seq = normalize_landmarks(seq)

        # Pad to MAX_SEQ_LEN=128
        T          = seq.shape[0]
        actual_len = min(T, 128)
        padded     = np.zeros((128, 130, 3), dtype=np.float32)
        padded[:actual_len] = seq[:actual_len]

        # To tensor: (1, 128, 130, 3)
        x     = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        s_len = torch.tensor([actual_len], dtype=torch.long).to(self.device)

        # Fingerspelling inference
        log_probs_fs = self.model(x, s_len, mode='fingerspelling')
        fs_decoded   = greedy_ctc_decode(
            log_probs_fs[0].cpu().numpy(),
            self.idx_to_char,
            self.blank_idx
        ).upper()

        # Signs inference
        logits_sg  = self.model(x, s_len, mode='signs')
        probs_sg   = torch.softmax(logits_sg[0], dim=-1).cpu().numpy()
        top5_idx   = probs_sg.argsort()[::-1][:5]
        top5_signs = [
            (self.idx_to_sign.get(int(i), f'SIGN_{i}').upper(),
             float(probs_sg[i]))
            for i in top5_idx
        ]

        result = {
            'fingerspelling': fs_decoded,
            'top_sign'      : top5_signs[0][0] if top5_signs else '',
            'top_sign_conf' : top5_signs[0][1] if top5_signs else 0.0,
            'top5_signs'    : top5_signs,
            'n_frames'      : actual_len,
            'ready'         : True,
        }

        self._last_result   = result
        self._last_n_frames = self._frame_counter
        return result

    def get_timing(self) -> float:
        """
        Return inference time in milliseconds for one predict() call.
        Returns 0.0 and prints a warning if the buffer is not yet full
        enough to run inference.

        Uses _force_rerun to bypass the cache without corrupting
        _last_n_frames.
        """
        if len(self.frame_buffer) < self.min_frames:
            print(f'⚠️  get_timing: buffer has {len(self.frame_buffer)} frames, '
                  f'need {self.min_frames} — returning 0.0 ms')
            return 0.0
        self._force_rerun = True
        t0 = time.perf_counter()
        self.predict()
        return (time.perf_counter() - t0) * 1000
