import cv2
import mediapipe as mp
import math
import winsound
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import numpy as np
import time
import pygame

pygame.mixer.init()
pygame.mixer.music.load("duck.mp3")


# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo MediaPipe
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange

# Hàm tính khoảng cách giữa hai điểm
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_v_sign(landmarks, image_width, image_height):
    # Lấy tọa độ cần thiết
    fingers = {
        "index_tip": landmarks[8],
        "index_dip": landmarks[7],
        "middle_tip": landmarks[12],
        "middle_dip": landmarks[11],
        "ring_tip": landmarks[16],
        "ring_dip": landmarks[15],
        "pinky_tip": landmarks[20],
        "pinky_dip": landmarks[19],
        "thumb_tip": landmarks[4],
        "thumb_ip": landmarks[3],
    }

    # Chuyển tọa độ keypoint từ tỷ lệ (0-1) sang pixel
    for key, landmark in fingers.items():
        fingers[key] = (int(landmark.x * image_width), int(landmark.y * image_height))
    
    # Kiểm tra hai ngón trỏ và giữa được duỗi ra (tip xa hơn DIP)
    index_straight = fingers["index_tip"][1] < fingers["index_dip"][1]
    middle_straight = fingers["middle_tip"][1] < fingers["middle_dip"][1]

    point = max(landmarks[6].y, landmarks[10].y)

    # Kiểm tra ngón cái, nhẫn và út gập lại (tip gần hơn DIP)
    thumb_folded = landmarks[4].y > point
    ring_folded = landmarks[16].y > point
    pinky_folded = landmarks[20].y > point

    # Khoảng cách giữa ngón trỏ và giữa đủ lớn
    v_distance = calculate_distance(fingers["index_tip"], fingers["middle_tip"])
    is_v = index_straight and middle_straight and thumb_folded and ring_folded and pinky_folded and v_distance > 50  # Điều chỉnh ngưỡng

    return is_v

# Hàm kiểm tra cử chỉ "Like"
def is_like_sign(landmarks, image_width, image_height):
    is_like = False
    if landmarks[3].y < min(landmarks[5].y, landmarks[9].y, landmarks[13].y, landmarks[17].y):
        is_like = True
    #nếu landmark[3].x không nằm giữa landmark[5].x và landmark[6].x
    if landmarks[3].x > landmarks[5].x and landmarks[3].x < landmarks[6].x:
        is_like = True
    else:
        is_like = False
    return is_like

def is_dislike_sign(landmarks, image_width, image_height):
    is_dislike = False

    if landmarks[3].y > max(landmarks[5].y, landmarks[9].y, landmarks[13].y, landmarks[17].y):
        is_dislike = True
    else:
        is_dislike = False

    if landmarks[2].x < landmarks[10].x and landmarks[2].x > landmarks[0].x:
        is_dislike = is_dislike and True
    else:
        is_dislike = False
    return is_dislike
    
def control_volume(landmarks, image_width, image_height):
    vertice_x, vertice_y = landmarks[0].x * image_width, landmarks[0].y * image_height
    thumb_x, thumb_y = landmarks[4].x * image_width, landmarks[4].y * image_height
    index_x, index_y = landmarks[8].x * image_width, landmarks[8].y * image_height

    vector_thumb = (thumb_x - vertice_x, thumb_y - vertice_y)
    vector_index = (index_x - vertice_x, index_y - vertice_y)
    
    dot_product = vector_thumb[0] * vector_index[0] + vector_thumb[1] * vector_index[1]
    magnitude_thumb = math.sqrt(vector_thumb[0]**2 + vector_thumb[1]**2)
    magnitude_index = math.sqrt(vector_index[0]**2 + vector_index[1]**2)

    theta = math.acos(dot_product / (magnitude_thumb * magnitude_index))
    degrees = math.degrees(theta)

    return degrees

# Khởi động video
cap = cv2.VideoCapture(0)

bool_v_sign = False
bool_adjust_volume = False
count_like_and_dislike = 0

bool_display_expected_volume = False

#Theo dõi frame và giá trị góc
frame_count = 0
stable_frame_count = 0
last_angle = None
angle_threshold = 5
current_time = time.time()
current_volume = volume.GetMasterVolumeLevel()

# Khởi tạo Mediapipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển sang RGB để xử lý
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Chuyển lại ảnh về BGR để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if bool_v_sign:
            cv2.putText(image, "On Mode Adjust Volume", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        else:   
            cv2.putText(image, "Off Mode Adjust Volume", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if bool_display_expected_volume and bool_v_sign:
            cv2.putText(image, "Expected Volume? Like if yes, Dislike if no", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ bàn tay và keypoints
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Lấy danh sách keypoints
                h, w, _ = image.shape
                landmarks = hand_landmarks.landmark

                # Kiểm tra cử chỉ "V"
                if is_v_sign(landmarks, w, h):
                    if not bool_v_sign:
                        cv2.putText(image, "V-Sign Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pass

                    bool_v_sign = True
                    bool_adjust_volume = True
                    current_time = time.time()

                if time.time() - current_time < 2: 
                    continue

                if bool_adjust_volume and bool_v_sign:
                    angle = control_volume(landmarks, w, h)
                    angle = int(angle/3) * 3
                    print("angle:, ", angle)
                    if angle > 55:
                        volume.SetMasterVolumeLevelScalar(1, None)
                    elif angle <= 6:
                        volume.SetMasterVolumeLevelScalar(0, None)
                    else:
                        vol = angle / 55.0
                        vol = round(vol,2)
                        volume.SetMasterVolumeLevelScalar(vol, None)
                    if last_angle is not None and abs(angle - last_angle) < angle_threshold:
                        stable_frame_count += 1
                    else:
                        stable_frame_count = 0
                    last_angle = angle

                    print(stable_frame_count)

                    if stable_frame_count >= 90:
                        bool_adjust_volume = False
                        stable_frame_count = 0
                        bool_display_expected_volume = True

                    # if stable_frame_count < 10:
                    #     if angle > 0.96:
                    #         volume.SetMasterVolumeLevelScalar(0, None)
                    #     elif angle < 0.6:
                    #         volume.SetMasterVolumeLevelScalar(1, None)
                    #     else:
                    #         vol = np.interp(angle, [0.6, 0.96], [maxVol, minVol])
                    #         volume.SetMasterVolumeLevel(vol, None)
                    # else:
                    #     bool_adjust_volume = False
                    #     cv2.putText(image, "Volume control disabled", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if bool_display_expected_volume:
                    if is_like_sign(landmarks, w, h):
                        count_like_and_dislike += 1
                    if is_dislike_sign(landmarks, w, h):
                        count_like_and_dislike -= 1

                    if count_like_and_dislike > 10:
                        cv2.putText(image, "Like Sign Detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        count_like_and_dislike = 0
                        bool_v_sign = False
                        bool_adjust_volume = False
                        bool_display_expected_volume = False
                        

                    if count_like_and_dislike < -10:
                        cv2.putText(image, "Dislike Sign Detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        count_like_and_dislike = 0
                        bool_v_sign = True
                        bool_adjust_volume = True
                        bool_display_expected_volume = False

                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pass
        # Hiển thị kết quả
        cv2.imshow('Hand Gesture Recognition', image)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #nếu tắt cửa sổ thì thoát
        if cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()