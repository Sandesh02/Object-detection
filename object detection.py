import cv2
import numpy as np

# Load YOLOv4 weights and configuration
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# Get the names of all the objects that YOLO can detect
with open("coconames", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the input size for the YOLOv4 model
input_size = (416, 416)

# Get the camera input
cap = cv2.VideoCapture(0)

# Set the minimum confidence level for detection
confidence_threshold = 0.5

# Set the non-maximum suppression threshold
nms_threshold = 0.4

# Define the color for the bounding boxes
box_color = (0, 255, 0)

# Loop through the frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the input size
    resized = cv2.resize(frame, input_size)

    # Normalize the pixel values of the input image
    normalized = resized / 255.0

    converted = cv2.convertScaleAbs(normalized, alpha=(255.0))

    # Convert image to a blob
    blob = cv2.dnn.blobFromImage(converted, scalefactor=1/255.0, size=input_size)

    # Set the input blob for the network
    net.setInput(blob)

    # Run forward pass to get output of the output layers
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []
    height, width, _ = frame.shape

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    for i in indices:

        i = int(i) if not isinstance(i, (list, tuple, np.ndarray)) else i[0]
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Display the output image
    cv2.imshow("Object Detection", frame)

    # Wait for key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()