import cv2
import numpy as np

# Đường dẫn đến file weights và config của mô hình YOLO
weights_path = "weights/yolov3.weights"
config_path = "cfg/yolov3.cfg"

# Đường dẫn đến file labels (tên của các lớp)
labels_path = "coco/coco.name"

# Đọc tên của các lớp từ file labels
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Load mô hình YOLOv3 và các trọng số được đào tạo trước
net = cv2.dnn.readNet(weights_path, config_path)

# Lấy chỉ số của các lớp mà mô hình YOLOv3 có thể phát hiện
layer_indexes = net.getUnconnectedOutLayers().flatten()
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in layer_indexes]

# Đọc ảnh đầu vào
image = cv2.imread("image/family2.jpg")

# Resize ảnh thành kích thước phù hợp với mô hình (YOLO sử dụng ảnh kích thước 416x416)
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Đưa ảnh vào mạng để phát hiện đối tượng
net.setInput(blob)
outs = net.forward(output_layers)

# Xác định các thông số của các hộp giới hạn và vẽ chúng lên ảnh
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Tính toán tọa độ của hộp giới hạn
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            # Lưu các thông số vào danh sách
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append((center_x, center_y, w, h))

# Áp dụng Non-maximum suppression để loại bỏ các hộp giới hạn trùng lặp
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Kiểm tra xem có đối tượng nào vượt qua ngưỡng
if len(indices) > 0:
    # Vẽ bounding box và in văn bản cho các đối tượng đã được phát hiện
    for i in indices.flatten():
        center_x, center_y, w, h = boxes[i]
        class_id = class_ids[i]
        label = f"{labels[class_id]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), color, 2)
        cv2.putText(image, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Hiển thị ảnh với các đối tượng đã được phát hiện
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()