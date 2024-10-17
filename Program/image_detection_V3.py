import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Text
from PIL import Image, ImageTk
import requests

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to fetch description and details from Wikipedia
def get_object_details(object_name):
    try:
        # Wikipedia API URL for searching summaries
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{object_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No description available.')
        else:
            return 'Description not found.'
    except Exception as e:
        return f'Error fetching details: {str(e)}'

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
    detected_objects = []
    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        # Draw rectangle around detected object
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Get the object name
        object_name = labels[int(class_id)]
        
        # Put label and confidence score
        label = f"{object_name}: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Fetch details for the object using Wikipedia API
        description = get_object_details(object_name)
        detected_objects.append((object_name, description))

    # Convert image back to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, detected_objects

# Function to browse and select an image
def browse_image():
    global img_label, text_box
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if file_path:
        # Run detection on the selected image
        detected_img, detected_objects = detect_image(file_path)
        
        # Convert to PIL image to display in Tkinter
        img = Image.fromarray(detected_img)
        img = img.resize((500, 400))  # Resize to fit in the window
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the image label in the GUI
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        
        # Clear previous text
        text_box.delete(1.0, tk.END)
        
        # Display object details in the text box
        for obj_name, obj_desc in detected_objects:
            text_box.insert(tk.END, f"Object: {obj_name}\nDescription: {obj_desc}\n\n")

# Initialize the GUI window
root = tk.Tk()
root.title("Object Detection with Metadata")
root.geometry("800x600")

# Add a label to display the selected image
img_label = Label(root)
img_label.pack()

# Add a button to browse and select an image
browse_btn = Button(root, text="Browse Image", command=browse_image)
browse_btn.pack(pady=20)

# Add a text box to display object details
text_box = Text(root, height=10, width=80)
text_box.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
