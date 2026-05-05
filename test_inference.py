# ============================================================
# test_inference.py
# Quick test of the inference pipeline without a webcam.
# Run this first to confirm everything is wired up correctly.
# Usage: python test_inference.py
# ============================================================

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference import ASLInferenceEngine

WEIGHTS_DIR  = Path('weights')
WEIGHTS_FILE = WEIGHTS_DIR / 'asl_model_weights_v2.pt'
VOCAB_FILE   = WEIGHTS_DIR / 'vocabulary.json'
SIGNS_FILE   = WEIGHTS_DIR / 'sign_to_idx.json'


def main():
    print('=== ASL Inference Test ===\n')

    # Check files
    missing = []
    for p in [WEIGHTS_FILE, VOCAB_FILE, SIGNS_FILE]:
        status = '✅' if p.exists() else '❌'
        size   = f'{p.stat().st_size/1e6:.1f} MB' if p.exists() else 'NOT FOUND'
        print(f'{status} {p.name:<35} {size}')
        if not p.exists():
            missing.append(str(p))

    if missing:
        print(f'\n❌ Missing files: {missing}')
        print('   Place model files in the weights/ folder.')
        sys.exit(1)

    print()

    # Load engine
    engine = ASLInferenceEngine(
        weights_path = str(WEIGHTS_FILE),
        vocab_path   = str(VOCAB_FILE),
        signs_path   = str(SIGNS_FILE),
        device       = 'cpu',
        buffer_size  = 64,
        min_frames   = 10,
    )

    print()

    # Test 1: Not enough frames
    print('--- Test 1: Empty buffer ---')
    result = engine.predict()
    print(f'Ready    : {result["ready"]}   (expected False)')
    print(f'N frames : {result["n_frames"]}   (expected 0)')

    # Test 2: Feed random landmark frames
    print('\n--- Test 2: Random landmark frames (20 frames) ---')
    engine.clear()
    for _ in range(20):
        # Random (130, 3) landmarks — not real signs but tests the pipeline
        fake_frame = np.random.randn(130, 3).astype(np.float32) * 0.1
        engine.add_frame(fake_frame)

    result = engine.predict()
    print(f'Ready          : {result["ready"]}   (expected True)')
    print(f'N frames       : {result["n_frames"]}')
    print(f'Fingerspelling : "{result["fingerspelling"]}"')
    print(f'Top sign       : {result["top_sign"]}  ({result["top_sign_conf"]*100:.1f}%)')
    print(f'Top 5 signs    :')
    for name, prob in result['top5_signs']:
        bar = '█' * int(prob * 30)
        print(f'  {name:<20} {prob*100:5.1f}%  {bar}')

    # Test 3: Inference timing
    print('\n--- Test 3: Inference speed ---')
    engine.clear()
    for _ in range(64):
        engine.add_frame(np.random.randn(130, 3).astype(np.float32) * 0.1)

    ms = engine.get_timing()
    fps_equiv = 1000 / ms if ms > 0 else 0
    print(f'Inference time  : {ms:.1f} ms')
    print(f'Equivalent FPS  : {fps_equiv:.0f}')

    if ms < 100:
        print('✅ Fast enough for real-time (< 100 ms)')
    elif ms < 200:
        print('⚠️  Borderline real-time (100–200 ms) — may feel slightly laggy')
    else:
        print('❌ Too slow for real-time — consider using GPU or reducing buffer_size')

    print('\n✅ All tests passed — run python run.py to start webcam')


if __name__ == '__main__':
    main()
