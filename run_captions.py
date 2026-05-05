# ============================================================
# run_captions.py
# Real-time ASL recognition from webcam — End-user caption view
#
# Caption style mirrors Google Voice / Live Captions:
#   - The current candidate sign appears immediately in grey (live preview)
#   - It turns white and is committed to the sentence once held long enough
#   - The sentence scrolls left on a single line as it grows
#
# Signs are CAPITALIZED.  Fingerspelling is prefixed "fs-" (e.g. fs-J).
# Usage: python run_captions.py
# ============================================================

from __future__ import annotations   # str | None on Python 3.9

import cv2
import time
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from inference import ASLInferenceEngine

try:
    from extract_landmarks import LandmarkExtractor130
    EXTRACTOR_AVAILABLE = True
except Exception as e:
    print(f'⚠️  Could not load LandmarkExtractor130: {e}')
    print('   Make sure mediapipe_models/ folder exists with .task files')
    EXTRACTOR_AVAILABLE = False


# ============================================================
# Config
# ============================================================
WEIGHTS_DIR   = Path('weights')
WEIGHTS_FILE  = WEIGHTS_DIR / 'asl_model_weights_v2.pt'
VOCAB_FILE    = WEIGHTS_DIR / 'vocabulary.json'
SIGNS_FILE    = WEIGHTS_DIR / 'sign_to_idx.json'

BUFFER_SIZE        = 64     # sliding window of frames
MIN_FRAMES         = 10     # minimum frames before predicting
DISPLAY_W          = 1280
DISPLAY_H          = 720

CONFIDENCE_THRESH  = 0.35   # minimum confidence to show a candidate at all
COMMIT_HOLD_FRAMES = 8      # frames a candidate must be stable before committing
CLEAR_AFTER_SEC    = 6.0    # seconds of silence before sentence auto-clears

SILENCE_TOKEN: Optional[str] = None   # None == no sign detected this frame


# ============================================================
# Gloss formatter  (inference result → display token or None)
# ============================================================
def format_token(result: dict) -> Optional[str]:
    """
    Priority:
      1. Fingerspelling decoded  →  "fs-<LETTERS>"
      2. Sign above threshold    →  "SIGN_NAME"
      3. Low confidence / not ready  →  None  (silence)
    """
    if not result['ready']:
        return SILENCE_TOKEN

    fs = result['fingerspelling'].strip().upper()
    if fs:
        return f'fs-{fs}'

    sign = result['top_sign']
    conf = result['top_sign_conf']
    if sign and conf >= CONFIDENCE_THRESH:
        return sign.upper()

    return SILENCE_TOKEN


# ============================================================
# Caption state  (Google Voice style)
# ============================================================
class CaptionState:
    """
    Tracks committed gloss tokens and the live in-progress candidate.

    Behaviour mirrors Google Voice:
      • As soon as a sign clears CONFIDENCE_THRESH it appears in grey
        as a live preview — immediately visible to the user.
      • After COMMIT_HOLD_FRAMES of stability it "locks in": turns white
        and is appended to the committed sentence.
      • Switching to a different sign resets the hold counter and updates
        the grey preview in real time.
      • After CLEAR_AFTER_SEC of silence the sentence auto-clears.
    """

    def __init__(self):
        self.committed: list[str]            = []
        self.live_token: Optional[str]       = None
        self._hold_count: int                = 0
        self._last_committed: Optional[str]  = None
        self._last_sign_time: float          = time.time()

    def update(self, token: Optional[str]) -> None:
        now = time.time()

        if token is not None:
            self._last_sign_time = now

        # Silence: clear preview, maybe auto-clear sentence
        if token is None:
            self.live_token  = None
            self._hold_count = 0
            if (now - self._last_sign_time) > CLEAR_AFTER_SEC and self.committed:
                self.committed.clear()
                self._last_committed = None
            return

        # Update live preview immediately so the user sees it right away
        self.live_token = token

        # Suppress re-committing the same token that was just locked in
        if token == self._last_committed:
            self._hold_count = 0
            return

        # Suppress duplicate of the last committed token in the sentence
        if self.committed and self.committed[-1] == token:
            self._hold_count = 0
            return

        # Accumulate hold count for stability check
        if self.live_token == token:
            self._hold_count += 1
        else:
            self._hold_count = 1

        # Commit once stable long enough
        if self._hold_count >= COMMIT_HOLD_FRAMES:
            self.committed.append(token)
            self._last_committed = token
            self._hold_count     = 0

    @property
    def hold_progress(self) -> float:
        """0.0–1.0 representing how close the live candidate is to committing."""
        return min(self._hold_count / COMMIT_HOLD_FRAMES, 1.0)

    def get_display_sentence(self) -> str:
        """Committed tokens as a single space-separated string."""
        return ' '.join(self.committed)

    def clear(self) -> None:
        self.committed.clear()
        self.live_token      = None
        self._hold_count     = 0
        self._last_committed = None
        self._last_sign_time = time.time()


# ============================================================
# Renderer constants
# ============================================================
FONT_MAIN  = cv2.FONT_HERSHEY_DUPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_PLAIN
FONT_UI    = cv2.FONT_HERSHEY_SIMPLEX

CAP_BAR_H   = 90    # total height of the caption strip
CAP_TEXT_Y  = 58    # text baseline measured from top of the bar
CAP_FS_MAIN = 1.3   # font scale for committed tokens
CAP_FS_LIVE = 1.1   # font scale for live preview token
CAP_THICK   = 2

COLOR_COMMITTED = (255, 255, 255)   # white  — locked tokens
COLOR_LIVE      = (160, 160, 160)   # grey   — live preview
COLOR_FS_COMMIT = (90,  200, 255)   # cyan   — committed fingerspelling
COLOR_FS_LIVE   = (60,  150, 200)   # dim cyan — live fingerspelling preview
COLOR_BG        = (12,   12,  12)   # near-black bar background
COLOR_HINT      = (55,   55,  55)   # dim hint text
COLOR_PROGRESS  = (100, 200, 100)   # green — hold-progress underline fill
COLOR_TRACK     = (55,   55,  55)   # grey  — hold-progress underline track


# ============================================================
# Caption bar renderer  (Google Voice style)
# ============================================================
def render_caption_bar(frame, caption: CaptionState) -> None:
    """
    Draw a single-line rolling caption bar at the bottom of the frame.

    The live preview token appears immediately in grey at the right edge of
    the committed text, with a green underline that fills as the hold counter
    counts up.  Once committed it turns white and the sentence scrolls left
    to make room for the next sign.
    """
    h, w = frame.shape[:2]
    bar_top = h - CAP_BAR_H

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, bar_top), (w, bar_top), (50, 50, 50), 1)

    y = bar_top + CAP_TEXT_Y

    # Hint text bottom-right
    cv2.putText(frame, 'Q quit  |  C clear',
                (w - 200, h - 10), FONT_PLAIN,
                0.95, COLOR_HINT, 1, cv2.LINE_AA)

    live = caption.live_token
    committed_str = caption.get_display_sentence()

    # Nothing to show yet — idle prompt
    if not committed_str and live is None:
        cv2.putText(frame, 'Begin signing\u2026',
                    (30, y), FONT_MAIN, 0.75, (70, 70, 70), 1, cv2.LINE_AA)
        return

    # ---- Measure live token so we can reserve space for it on the right ----
    LEFT_MARGIN  = 30
    RIGHT_MARGIN = 30
    live_w = 0
    if live is not None:
        (live_w, _), _ = cv2.getTextSize(live, FONT_MAIN, CAP_FS_LIVE, 1)
        live_w += 20   # gap between last committed token and live preview

    # Available width for committed tokens
    avail_w = w - LEFT_MARGIN - RIGHT_MARGIN - live_w

    # Build the list of committed tokens that fit, keeping the most recent ones
    # (older tokens drop off the left edge — scroll behaviour)
    tokens = committed_str.split(' ') if committed_str else []
    display_tokens: list[str] = []
    used_w = 0
    for tok in reversed(tokens):
        (tw, _), _ = cv2.getTextSize(tok, FONT_MAIN, CAP_FS_MAIN, CAP_THICK)
        gap = 18
        if used_w + tw + gap > avail_w and display_tokens:
            break   # this token doesn't fit — older tokens are scrolled off
        display_tokens.insert(0, tok)
        used_w += tw + gap

    # Draw committed tokens left-to-right
    x = LEFT_MARGIN
    for tok in display_tokens:
        color = COLOR_FS_COMMIT if tok.startswith('fs-') else COLOR_COMMITTED
        cv2.putText(frame, tok, (x, y), FONT_MAIN, CAP_FS_MAIN,
                    color, CAP_THICK, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(tok, FONT_MAIN, CAP_FS_MAIN, CAP_THICK)
        x += tw + 18

    # Draw live preview token
    if live is not None:
        live_color = COLOR_FS_LIVE if live.startswith('fs-') else COLOR_LIVE
        cv2.putText(frame, live, (x, y), FONT_MAIN, CAP_FS_LIVE,
                    live_color, 1, cv2.LINE_AA)

        # Hold-progress underline beneath the live token
        (lw, _), _ = cv2.getTextSize(live, FONT_MAIN, CAP_FS_LIVE, 1)
        progress_w  = int(lw * caption.hold_progress)
        underline_y = y + 7
        # Track (dim full-width line)
        cv2.line(frame, (x, underline_y), (x + lw, underline_y),
                 COLOR_TRACK, 2)
        # Fill (green portion that grows as hold count increases)
        if progress_w > 0:
            cv2.line(frame, (x, underline_y), (x + progress_w, underline_y),
                     COLOR_PROGRESS, 2)


# ============================================================
# Main loop
# ============================================================
def main():
    missing = [str(p) for p in [WEIGHTS_FILE, VOCAB_FILE, SIGNS_FILE]
               if not p.exists()]
    if missing:
        print('❌ Missing required files:')
        for m in missing:
            print(f'   {m}')
        print('\nMake sure all files are in the weights/ folder.')
        sys.exit(1)

    engine = ASLInferenceEngine(
        weights_path = str(WEIGHTS_FILE),
        vocab_path   = str(VOCAB_FILE),
        signs_path   = str(SIGNS_FILE),
        device       = 'cpu',
        buffer_size  = BUFFER_SIZE,
        min_frames   = MIN_FRAMES,
    )

    ms = engine.get_timing()
    if ms > 0:
        print(f'Inference time : {ms:.1f} ms per frame')

    extractor = None
    if EXTRACTOR_AVAILABLE:
        try:
            extractor = LandmarkExtractor130()
            print('Landmark extractor ready ✅')
        except Exception as e:
            print(f'⚠️  Extractor init failed: {e}')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Cannot open webcam')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    print('✅ Webcam open — press Q to quit, C to clear captions')

    caption               = CaptionState()
    extractor_error_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            if extractor is not None:
                try:
                    lm = extractor.extract(frame)
                    engine.add_frame(lm)
                    extractor_error_count = 0
                except Exception as e:
                    extractor_error_count += 1
                    if extractor_error_count % 30 == 1:
                        print(f'⚠️  Extractor error (frame skipped): {e}')

            result = engine.predict()
            token  = format_token(result)
            caption.update(token)

            render_caption_bar(frame, caption)

            if extractor is None:
                cv2.putText(frame,
                            'MediaPipe not available — no landmarks extracted',
                            (20, 40), FONT_UI,
                            0.6, (0, 80, 200), 2, cv2.LINE_AA)

            cv2.imshow('ASL Captions', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                engine.clear()
                caption.clear()
                print('Captions cleared')
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print('Done')


if __name__ == '__main__':
    main()
