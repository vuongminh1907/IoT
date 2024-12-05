import cv2
import mediapipe as mp
import math

# Khởi tạo MediaPipe
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

# Khởi động video
cap = cv2.VideoCapture(0)

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ bàn tay và keypoints
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Lấy danh sách keypoints
                h, w, _ = image.shape
                landmarks = hand_landmarks.landmark

                # Kiểm tra cử chỉ "V"
                if is_v_sign(landmarks, w, h):
                    cv2.putText(image, "V-Sign Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow('Hand Gesture Recognition', image)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
