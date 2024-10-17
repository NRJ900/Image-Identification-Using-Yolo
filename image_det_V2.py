import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to load and detect objects in the selected image
def detect_image(image_path):
    # Load the image
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

    # Convert image back to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Function to browse and select an image
def browse_image():
    global img_label
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if file_path:
        # Run detection on the selected image
        detected_img = detect_image(file_path)
        
        # Convert to PIL image to display in Tkinter
        img = Image.fromarray(detected_img)
        img = img.resize((500, 400))  # Resize to fit in the window
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the image label in the GUI
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection

# Initialize the GUI window
root = tk.Tk()
root.title("Object Detection")
root.geometry("600x500")

# Add a label to display the selected image
img_label = Label(root)
img_label.pack()

# Add a button to browse and select an image
browse_btn = Button(root, text="Browse Image", command=browse_image)
browse_btn.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
