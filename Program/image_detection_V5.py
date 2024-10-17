#There are several problems in this program,eg.not describing the image completely

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
def get_wikipedia_details(object_name):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{object_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No description available.')
        else:
            return 'Description not found.'
    except Exception as e:
        return f'Error fetching Wikipedia details: {str(e)}'

# Function to fetch structured data from Wikidata
def get_wikidata_details(object_name):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={object_name}&language=en&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['search']:
                entity_id = data['search'][0]['id']
                detail_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                detail_response = requests.get(detail_url)
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    entity = detail_data['entities'][entity_id]
                    description = entity.get('descriptions', {}).get('en', {}).get('value', 'No description available.')
                    return description
        return 'No detailed information found on Wikidata.'
    except Exception as e:
        return f'Error fetching Wikidata details: {str(e)}'

# Function to fetch biological classification from iNaturalist
def get_taxonomy_details(object_name):
    try:
        url = f"https://api.inaturalist.org/v1/taxa?q={object_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                taxon = data['results'][0]
                scientific_name = taxon.get('name', 'Unknown')
                rank = taxon.get('rank', 'Unknown')
                return f"Scientific Name: {scientific_name}, Rank: {rank}"
        return 'No taxonomy information found.'
    except Exception as e:
        return f'Error fetching taxonomy: {str(e)}'

# Function to load and detect objects in the selected image
def detect_image(image_path=None, image_frame=None):
    if image_path:
        # Load the image from the path
        image = cv2.imread(image_path)
    elif image_frame is not None:
        # Use the captured frame from the camera
        image = image_frame
    else:
        return None, []

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

        # Fetch details from Wikipedia, Wikidata, and iNaturalist API
        wiki_desc = get_wikipedia_details(object_name)
        wikidata_desc = get_wikidata_details(object_name)
        taxonomy = get_taxonomy_details(object_name)

        detected_objects.append((object_name, wiki_desc, wikidata_desc, taxonomy))

    # Convert image back to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, detected_objects

# Function to browse and select an image
def browse_image():
    global img_label, text_box
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if file_path:
        detected_img, detected_objects = detect_image(image_path=file_path)
        update_image_and_details(detected_img, detected_objects)

# Function to capture image from the webcam
def capture_from_camera():
    global img_label, text_box
    video_capture = cv2.VideoCapture(0)

    if video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            detected_img, detected_objects = detect_image(image_frame=frame)
            update_image_and_details(detected_img, detected_objects)

    # Release the camera after capture
    video_capture.release()

# Function to update the image and object details in the GUI
def update_image_and_details(detected_img, detected_objects):
    if detected_img is not None:
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
        for obj_name, wiki_desc, wikidata_desc, taxonomy in detected_objects:
            text_box.insert(tk.END, f"Object: {obj_name}\n")
            text_box.insert(tk.END, f"Wikipedia: {wiki_desc}\n")
            text_box.insert(tk.END, f"Wikidata: {wikidata_desc}\n")
            text_box.insert(tk.END, f"Taxonomy: {taxonomy}\n\n")

# Initialize the GUI window
root = tk.Tk()
root.title("Object Detection with Detailed Information")
root.geometry("800x600")

# Add a label to display the selected image
img_label = Label(root)
img_label.pack()

# Add buttons for browsing and capturing from camera
browse_btn = Button(root, text="Browse Image", command=browse_image)
browse_btn.pack(pady=10)

capture_btn = Button(root, text="Capture from Camera", command=capture_from_camera)
capture_btn.pack(pady=10)

# Add a text box to display object details
text_box = Text(root, height=10, width=80)
text_box.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
