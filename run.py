# ============================================================
# run.py
# Real-time ASL recognition from webcam — Developer view
# Shows landmarks, confidence bars, top-5 signs, FPS, frame count
# Usage: python run.py
# ============================================================

import cv2
import numpy as np
import time
import sys
from pathlib import Path

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

BUFFER_SIZE   = 64    # sliding window of frames
MIN_FRAMES    = 10    # minimum before predicting
DISPLAY_W     = 1280
DISPLAY_H     = 720
FPS_SMOOTH    = 10    # frames to average FPS over


# ============================================================
# Display helpers
# ============================================================
def draw_ui(frame, result, fps, extractor_ok, min_frames):
    """Draw the inference overlay on the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 200), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = h - 180

    if not extractor_ok:
        cv2.putText(frame, 'MediaPipe models not found — check mediapipe_models/ folder',
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return

    if not result['ready']:
        cv2.putText(frame,
                    f'Collecting frames... ({result["n_frames"]}/{min_frames} needed)',
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return

    # Fingerspelling
    cv2.putText(frame, 'Fingerspelling:', (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    fs_text = result['fingerspelling'] if result['fingerspelling'] else '(silence)'
    cv2.putText(frame, fs_text, (200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 100), 2)

    y += 35

    # Top sign
    sign_name = result['top_sign']
    sign_conf = result['top_sign_conf']
    conf_pct  = int(sign_conf * 100)

    cv2.putText(frame, 'Top sign:', (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    if conf_pct >= 60:
        color = (100, 220, 100)
    elif conf_pct >= 35:
        color = (100, 200, 220)
    else:
        color = (150, 150, 150)

    cv2.putText(frame, f'{sign_name}  ({conf_pct}%)', (200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    y += 35

    # Top 3 signs bar
    cv2.putText(frame, 'Top 3:', (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    x_bar = 100
    for sign, prob in result['top5_signs'][:3]:
        bar_w = int(prob * 180)
        cv2.rectangle(frame, (x_bar, y - 14), (x_bar + bar_w, y), (70, 130, 70), -1)
        cv2.putText(frame, f'{sign} {int(prob*100)}%',
                    (x_bar + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        x_bar += 300

    y += 35

    cv2.putText(frame,
                f'Frames: {result["n_frames"]}  |  FPS: {fps:.0f}  |  Press Q to quit',
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


def draw_landmarks(frame, lm):
    """Draw 130 landmarks on the frame (face, pose, hands)."""
    h, w = frame.shape[:2]

    # Face: 0–76  (green)
    for lx, ly, _ in lm[0:76]:
        if lx != 0.0 or ly != 0.0:
            cv2.circle(frame, (int(lx * w), int(ly * h)), 2, (0, 255, 0), -1)

    # Pose: 76–88  (blue)
    for lx, ly, _ in lm[76:88]:
        if lx != 0.0 or ly != 0.0:
            cv2.circle(frame, (int(lx * w), int(ly * h)), 2, (255, 0, 0), -1)

    # Left hand: 88–109  (red)
    for lx, ly, _ in lm[88:109]:
        if lx != 0.0 or ly != 0.0:
            cv2.circle(frame, (int(lx * w), int(ly * h)), 2, (0, 0, 255), -1)

    # Right hand: 109–130  (orange)
    for lx, ly, _ in lm[109:130]:
        if lx != 0.0 or ly != 0.0:
            cv2.circle(frame, (int(lx * w), int(ly * h)), 2, (0, 165, 255), -1)


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
        print('See README.md for the full folder structure.')
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
    print('✅ Webcam open — press Q to quit, C to clear buffer')

    fps_times = []

    try:
        while True:
            t0  = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if extractor is not None:
                try:
                    lm = extractor.extract(frame)
                    engine.add_frame(lm)
                    draw_landmarks(frame, lm)
                except Exception:
                    pass

            result = engine.predict()

            fps_times.append(time.perf_counter() - t0)
            if len(fps_times) > FPS_SMOOTH:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            draw_ui(frame, result, fps, extractor is not None, engine.min_frames)

            cv2.imshow('ASL Recognition — Developer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                engine.clear()
                print('Buffer cleared')
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print('Done')


if __name__ == '__main__':
    main()
