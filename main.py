import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            right_landmark_list = get_right_hand_landmarks(frame, processed, draw, mpHands)
            # Change volume using the right hand
            if right_landmark_list:
                right_distance = get_distance(frame, right_landmark_list)
                vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def get_right_hand_landmarks(frame, processed, draw, mpHands):
    right_landmark_list = []

    if processed.multi_hand_landmarks:
        for idx, handlm in enumerate(processed.multi_hand_landmarks):
            for landmark_idx, found_landmark in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(found_landmark.x * width), int(found_landmark.y * height)
                if landmark_idx == 4 or landmark_idx == 8:
                    right_landmark_list.append([landmark_idx, x, y])

            # Draw hand landmarks
            if idx == 0:  # Assume the first detected hand is the right hand
                draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return right_landmark_list


def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), \
        (landmark_list[1][1], landmark_list[1][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    distance = hypot(x2 - x1, y2 - y1)

    return distance


if __name__ == '__main__':
    main()
