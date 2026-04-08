# -*- coding: utf-8 -*-

import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-10
    cosine_angle = np.dot(ba, bc) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def get_landmark_xy(landmarks, landmark_enum):
    landmark = landmarks[landmark_enum.value]
    return [landmark.x, landmark.y], landmark.visibility


def get_best_leg(landmarks):
    left_hip, left_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    left_knee, left_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    left_ankle, left_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)

    right_hip, right_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    right_knee, right_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    right_ankle, right_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    left_score = left_hip_vis + left_knee_vis + left_ankle_vis
    right_score = right_hip_vis + right_knee_vis + right_ankle_vis

    if left_score >= right_score:
        return left_hip, left_knee, left_ankle, min(left_hip_vis, left_knee_vis, left_ankle_vis), "left"
    else:
        return right_hip, right_knee, right_ankle, min(right_hip_vis, right_knee_vis, right_ankle_vis), "right"


def classify_squat(knee_angle):
    if knee_angle > 165:
        return "Standing"
    elif 130 < knee_angle <= 165:
        return "Descending"
    elif 90 <= knee_angle <= 130:
        return "Parallel (good)"
    elif 70 <= knee_angle < 90:
        return "Below parallel"
    elif 50 <= knee_angle < 70:
        return "Deep squat"
    else:
        return "Very deep / unstable"


def update_rep_count(knee_angle, stage, counter):
    """
    Count 1 rep when user goes:
    standing -> squat depth -> standing
    """
    if knee_angle > 165:
        if stage == "down":
            counter += 1
        stage = "up"
    elif knee_angle < 130:
        stage = "down"

    return stage, counter


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or np.isnan(fps):
        fps = 20.0

    videos_dir = "saved_videos"
    os.makedirs(videos_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(videos_dir, f"squat_output_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    counter = 0
    stage = "up"

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            status = "Move back so full body is visible"
            knee_angle = None
            side_used = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                hip_point, knee_point, ankle_point, leg_vis, side_used = get_best_leg(landmarks)

                if leg_vis > 0.3:
                    knee_angle = calculate_angle(hip_point, knee_point, ankle_point)
                    status = classify_squat(knee_angle)
                    stage, counter = update_rep_count(knee_angle, stage, counter)

                    h, w, _ = frame.shape
                    knee_px = tuple(np.multiply(knee_point, [w, h]).astype(int))

                    cv2.putText(
                        frame,
                        f"{side_used}: {int(knee_angle)} deg",
                        knee_px,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Top info panel
            cv2.rectangle(frame, (10, 10), (470, 140), (0, 0, 0), -1)

            cv2.putText(frame, "Squat Form Checker", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Status: {status}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if knee_angle is not None:
                cv2.putText(frame, f"Knee angle: {int(knee_angle)}", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if side_used is not None:
                cv2.putText(frame, f"Leg used: {side_used}", (250, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Reps: {counter}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Stage: {stage}", (180, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out.write(frame)

            cv2.imshow("Squat Form Checker", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved video: {video_filename}")


if __name__ == "__main__":
    main()