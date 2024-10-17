import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
image_path = 'F:\watch.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert image to RGB (YOLOv5 expects RGB images)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
results = model(rgb_image)

# Extract the detection results
detections = results.xyxy[0].cpu().numpy()  # xyxy format (x_min, y_min, x_max, y_max, confidence, class)

# Load class labels
labels = model.names

# Draw the detection results on the original image
for detection in detections:
    x_min, y_min, x_max, y_max, confidence, class_id = detection
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    # Draw rectangle around detected object
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Put label and confidence score
    label = f"{labels[int(class_id)]}: {confidence:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Show the image with detections
cv2.imshow('Image Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
