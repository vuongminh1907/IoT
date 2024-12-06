import cv2
import mediapipe as mp
import numpy as np
import math
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# Khởi tạo MediaPipe
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

def control_volume(landmarks, image_width, image_height):
    vertice_x, vertice_y = landmarks[0]
    thumb_x, thumb_y = landmarks[4]
    index_x, index_y = landmarks[8]

    vector_thumb = (thumb_x - vertice_x, thumb_y - vertice_y)
    vector_index = (index_x - vertice_x, index_y - vertice_y)
    
    dot_product = vector_thumb[0] * vector_index[0] + vector_thumb[1] * vector_index[1]
    magnitude_thumb = math.sqrt(vector_thumb[0]**2 + vector_thumb[1]**2)
    magnitude_index = math.sqrt(vector_index[0]**2 + vector_index[1]**2)

    cos_theta = dot_product / (magnitude_thumb * magnitude_index)

    return cos_theta, (thumb_x, thumb_y), (index_x, index_y), (vertice_x, vertice_y)

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


# Khởi động video
cap = cv2.VideoCapture(0)

bool_v_sign = False
bool_adjust_volume = False
count_like_and_dislike = 0

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ bàn tay và keypoints
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Lấy danh sách keypoints
                h, w, _ = image.shape
                landmarks = hand_landmarks.landmark

                angle, thumb, index, vertice = control_volume([(landmark.x * w, landmark.y * h) for landmark in landmarks], w, h)
                cv2.line(image, (int(thumb[0]), int(thumb[1])), (int(vertice[0]), int(vertice[1])), (255, 0, 0), 2)
                cv2.line(image, (int(index[0]), int(index[1])), (int(vertice[0]), int(vertice[1])), (255, 0, 0), 2)
                
                cv2.putText(image, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if angle > 0.96:
                    volume.SetMasterVolumeLevelScalar(0, None)
                elif angle < 0.6:
                    volume.SetMasterVolumeLevelScalar(1, None)
                else:
                    vol = np.interp(angle, [0.6, 0.96], [maxVol, minVol])
                    volume.SetMasterVolumeLevel(vol, None)
                # Kiểm tra cử chỉ "V"
                if is_v_sign(landmarks, w, h):
                    if not bool_v_sign:
                        cv2.putText(image, "V-Sign Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    bool_v_sign = True
                    bool_adjust_volume = True

                if bool_adjust_volume:
                    print("Adjust Volume")

                if is_like_sign(landmarks, w, h):
                    count_like_and_dislike += 1
                if is_dislike_sign(landmarks, w, h):
                    count_like_and_dislike -= 1

                if count_like_and_dislike > 10:
                    cv2.putText(image, "Like Sign Detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    count_like_and_dislike = 0
                    bool_v_sign = False

                if count_like_and_dislike < -10:
                    cv2.putText(image, "Dislike Sign Detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    count_like_and_dislike = 0
                    bool_v_sign = False
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
