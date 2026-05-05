# ============================================================
# extract_landmarks.py
# - Extracts exactly 130 landmarks per frame
# - Matches the Kaggle ASL competition winning-solution standard
#
# Layout  [0:76]   face  — lips, nose, left eye, right eye, brows
#         [76:88]  pose  — upper body (shoulders, elbows, wrists)
#         [88:109] left  hand — 21 pts
#         [109:130] right hand — 21 pts
#
# Can be IMPORTED or RUN DIRECTLY
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# MediaPipe model paths
# ============================================================
HAND_MODEL = "mediapipe_models/hand_landmarker.task"
POSE_MODEL = "mediapipe_models/pose_landmarker_full.task"
FACE_MODEL = "mediapipe_models/face_landmarker.task"

# ============================================================
# Initialize MediaPipe Tasks
# ============================================================
BaseOptions = python.BaseOptions

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2
    )
)

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        output_segmentation_masks=False
    )
)

face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
)

# ============================================================
# 130-Landmark Index Definitions
# Matches the Kaggle ASL competition winning-solution standard
# ============================================================

# --- LIPS (40 landmarks) ---
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409
]
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308, 191, 80, 81, 82, 13,
    312, 311, 310, 415
]
LIPS_IDXS = LIPS_OUTER + LIPS_INNER  # 40 landmarks

# --- NOSE (2 landmarks) ---
NOSE_IDXS = [1, 4]

# --- LEFT EYE (8 landmarks) ---
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144, 145, 154]

# --- RIGHT EYE (8 landmarks) ---
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380, 374, 381]

# --- LEFT EYEBROW (9 landmarks) ---
LEFT_EYEBROW_IDXS = [70, 63, 105, 66, 107, 55, 65, 52, 53]

# --- RIGHT EYEBROW (9 landmarks) ---
RIGHT_EYEBROW_IDXS = [336, 296, 334, 293, 300, 285, 295, 282, 283]

# All face landmarks combined: 40 + 2 + 8 + 8 + 9 + 9 = 76
FACE_IDXS = (
    LIPS_IDXS +
    NOSE_IDXS +
    LEFT_EYE_IDXS +
    RIGHT_EYE_IDXS +
    LEFT_EYEBROW_IDXS +
    RIGHT_EYEBROW_IDXS
)

# --- POSE: 12 upper-body landmarks ---
POSE_IDXS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

NUM_HAND = 21
NUM_FACE = len(FACE_IDXS)    # 76
NUM_POSE = len(POSE_IDXS)    # 12
TOTAL_LANDMARKS = NUM_FACE + NUM_POSE + 2 * NUM_HAND  # 130

assert TOTAL_LANDMARKS == 130, f"Expected 130 landmarks, got {TOTAL_LANDMARKS}"


# ============================================================
# Landmark Extractor Class
# ============================================================
class LandmarkExtractor130:
    """
    Extracts the competition-standard 130-landmark set per frame.

    Output layout (shape: (130, 3)):
        [0   : 76 ]  face  — lips(40) + nose(2) + eyes(16) + brows(18)
        [76  : 88 ]  pose  — upper body (shoulders, elbows, wrists)
        [88  : 109]  left  hand — 21 MediaPipe hand landmarks
        [109 : 130]  right hand — 21 MediaPipe hand landmarks

    Missing detections are returned as zeros (NaN-free for PyTorch use).
    """

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(mp.ImageFormat.SRGB, frame_rgb)

        # ---- Face ------------------------------------------------
        face_result = face_detector.detect(mp_image)
        if face_result.face_landmarks:
            full_face = face_result.face_landmarks[0]
            face = np.array(
                [[full_face[i].x, full_face[i].y, full_face[i].z]
                 for i in FACE_IDXS],
                dtype=np.float32
            )
        else:
            face = np.zeros((NUM_FACE, 3), dtype=np.float32)

        # ---- Pose (upper body only) ------------------------------
        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            all_pose = pose_result.pose_landmarks[0]
            pose = np.array(
                [[all_pose[i].x, all_pose[i].y, all_pose[i].z]
                 for i in POSE_IDXS],
                dtype=np.float32
            )
        else:
            pose = np.zeros((NUM_POSE, 3), dtype=np.float32)

        # ---- Hands -----------------------------------------------
        # Slot 0 = left hand, Slot 1 = right hand (fixed slots)
        hand_result = hand_detector.detect(mp_image)
        left_hand   = np.zeros((NUM_HAND, 3), dtype=np.float32)
        right_hand  = np.zeros((NUM_HAND, 3), dtype=np.float32)

        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks[:2]):
                label = hand_result.handedness[i][0].category_name  # 'Left' or 'Right'
                pts   = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms],
                    dtype=np.float32
                )
                if label == "Left":
                    left_hand = pts
                else:
                    right_hand = pts

        # ---- Stack in competition order --------------------------
        landmarks = np.vstack([face, pose, left_hand, right_hand])  # (130, 3)
        return landmarks


# ============================================================
# Webcam visualizer (standalone mode)
# ============================================================
def _draw_points(frame, points, color, radius=2):
    h, w, _ = frame.shape
    for x, y, _ in points:
        if x != 0.0 or y != 0.0:
            cv2.circle(frame, (int(x * w), int(y * h)), radius, color, -1)


def run_webcam():
    cap       = cv2.VideoCapture(0)
    extractor = LandmarkExtractor130()

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print(f"Webcam running — extracting {TOTAL_LANDMARKS} landmarks per frame")
    print("   Face(76) | Pose(12) | Left hand(21) | Right hand(21)")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        landmarks = extractor.extract(frame)

        i          = 0
        face       = landmarks[i:i+NUM_FACE];  i += NUM_FACE
        pose       = landmarks[i:i+NUM_POSE];  i += NUM_POSE
        left_hand  = landmarks[i:i+NUM_HAND];  i += NUM_HAND
        right_hand = landmarks[i:i+NUM_HAND]

        _draw_points(frame, face,       (0,   255,   0))   # green  — face
        _draw_points(frame, pose,       (255,   0,   0))   # blue   — pose
        _draw_points(frame, left_hand,  (0,     0, 255))   # red    — left hand
        _draw_points(frame, right_hand, (0,   165, 255))   # orange — right hand

        cv2.putText(frame,
                    "130 landmarks | Face:76 Pose:12 Hands:21+21",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ASL 130-Landmark Viewer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")


def main():
    run_webcam()


if __name__ == "__main__":
    main()