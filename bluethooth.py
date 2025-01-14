import serial
import serial.tools.list_ports

def find_bluetooth_port(baudrate=9600, timeout=1):
    # Liệt kê tất cả các cổng COM khả dụng
    ports = serial.tools.list_ports.comports()
    print("Đang kiểm tra các cổng COM khả dụng...")

    for port in ports:
        print(f"Đang thử cổng: {port.device}")
        try:
            # Thử kết nối đến cổng với cấu hình Bluetooth
            ser = serial.Serial(port=port.device, baudrate=baudrate, timeout=timeout)
            ser.write(b"AT\r\n")  # Gửi lệnh AT để kiểm tra phản hồi
            response = ser.readline().decode().strip()
            ser.close()

            # Kiểm tra nếu thiết bị phản hồi
            if response:
                print(f"Phản hồi từ thiết bị trên {port.device}: {response}")
                return port.device
        except Exception as e:
            print(f"Lỗi khi kiểm tra {port.device}: {e}")

    print("Không tìm thấy thiết bị Bluetooth nào thỏa mãn.")
    return None

# Chạy hàm tìm kiếm
bluetooth_port = find_bluetooth_port()

if bluetooth_port:
    print(f"Đã tìm thấy thiết bị Bluetooth trên cổng: {bluetooth_port}")
else:
    print("Không tìm thấy cổng Bluetooth hợp lệ.")
