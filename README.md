# Real-Time Squat Form Checker Using Computer Vision

## Overview
This project is a real-time squat form checker built with computer vision. Using a webcam feed, the system detects body landmarks, calculates knee angle, classifies squat depth, and counts repetitions in real time.

The goal of this project was to explore how pose-based geometry can support exercise feedback without wearable sensors.

## Features
- Real-time pose estimation using MediaPipe
- Knee angle calculation from hip-knee-ankle landmarks
- Squat depth feedback
- Rep counting
- Processed video recording with saved outputs

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works
1. Capture webcam video
2. Detect body pose landmarks
3. Select the more visible leg
4. Compute knee angle
5. Classify squat depth
6. Count a rep when the movement goes from standing to squat depth and back to standing
7. Save the processed video output

## Squat States
The current version uses approximate knee-angle thresholds:
- Standing: > 165°
- Descending: 130° to 165°
- Parallel (good): 90° to 130°
- Below parallel: 70° to 90°
- Deep squat: 50° to 70°

These values were calibrated empirically based on observed webcam output and may vary depending on camera position, body proportions, and pose-estimation noise.

![alt text](https://github.com/timothyhaminjo/Real-Time-Squat-Form-Checker-/blob/main/Screenshot1.png) 

![alt text](https://github.com/timothyhaminjo/Real-Time-Squat-Form-Checker-/blob/main/Screenshot2.png)

## Why This Matters
This project demonstrates a simple real-time biomechanical feedback system using computer vision. While implemented for squat analysis, the same general pipeline can be extended to other motion-feedback applications that require real-time angle estimation and performance guidance.

## File Structure
```text
squat-form-checker/
├── main.py
├── README.md
└── saved_videos/
