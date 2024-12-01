import cv2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

class_names_file = 'coco.names'
try:
    with open(class_names_file, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
except FileNotFoundError:
    print(f"Error: File '{class_names_file}' not found.")
    exit(1)

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

try:
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except cv2.error as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Press 'q' to quit.")
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5)

    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)

            label = f"{class_names[class_id - 1].upper()} {confidence:.2f}"
            cv2.putText(frame, label, (box[0] + 10, box[1] + 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
