import cv2
import numpy as np
import onnxruntime as ort

def preprocess(image, input_size):
    """
    Tiền xử lý ảnh để phù hợp với model ONNX.
    - Resize ảnh.
    - Chuẩn hóa giá trị pixel.

    Args:
        image (np.ndarray): Ảnh đầu vào.
        input_size (tuple): Kích thước đầu vào của model (width, height).

    Returns:
        np.ndarray: Dữ liệu đã được tiền xử lý.
    """
    resized = cv2.resize(image, input_size)
    #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    transposed = np.transpose(resized, (2, 0, 1))  # Chuyển sang định dạng (C, H, W)
    transposed = transposed.astype(np.float32) / 255.0
    return np.expand_dims(transposed, axis=0)  # Thêm batch dimension


def postprocess(output):
    """
    Xử lý kết quả từ model.

    Args:
        output (np.ndarray): Kết quả inference từ ONNX model.

    Returns:
        dict: Dictionary chứa các tọa độ landmark.
    """
    landmarks = output.reshape(-1, 3)  # [num_landmarks, (x, y, z)]
    return landmarks


# Đường dẫn đến model và kích thước đầu vào
model_path = "./model/MediaPipeHandLandmarkDetector.onnx"
input_size = (256, 256)  # Giả định kích thước đầu vào của model là 224x224

# Load model ONNX
session = ort.InferenceSession(model_path)

# Tên đầu vào và đầu ra
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[2].name

img = cv2.imread("tay.png")
img = cv2.resize(img, (256, 256))


# Tiền xử lý ảnh
input_data = preprocess(img, input_size)

# Chạy inference
outputs = session.run([output_name], {input_name: input_data})
print(outputs[0])


#draw landmark
for landmark in outputs[0][0]:
    x, y, z = landmark
    cv2.circle(img, (int(x * 256), int(y * 256)), 5, (0, 255, 0), -1)

cv2.imwrite("result.png", img)