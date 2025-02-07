# 🚀 Image Identification Using YOLOv5  

This project implements a **basic image identification module** in Python using **YOLOv5**. It consists of **five versions**, ranging from basic to intermediate-level programs, designed to help understand **object detection** using YOLOv5.  

📌 **YOLOv5 Repository**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  

---

## 🛠 Steps to Install  

### **1️⃣ Install Git on Your System**  

#### **For Windows**:  
1. **Download Git:**  
   - Visit [Git for Windows](https://git-scm.com/downloads/win) and download the installer.  
   - Alternatively, open **CMD** and run:  
     ```sh
     winget install --id Git.Git -e --source winget
     ```
2. **Install Git:**  
   - Run the installer and select **"Add Git to the system PATH"** (ensures Git is accessible via terminal).  

### **2️⃣ Clone YOLOv5 Repository**  
Once Git is installed, clone the **YOLOv5 repository** using:  
```sh
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

### **3️⃣ Install Required Modules**  
Open **CMD** and install dependencies:  
```sh
pip install torch torchvision opencv-python pillow requests
```

---

## 🏃 Run the Program  
After installing dependencies, run the YOLOv5 script to detect objects in an image:  
```sh
python detect.py --source image.jpg --weights yolov5s.pt --conf 0.4
```
- Replace `image.jpg` with your **image file path**.  
- Adjust `--conf 0.4` for confidence threshold tuning.  

---

## 📌 Features  
✔️ Basic to intermediate **image identification** using YOLOv5.  
✔️ **Step-by-step setup guide** for easy installation.  
✔️ Uses **pre-trained YOLOv5 models** for object detection.  

---


## 🔗 References  
- **YOLOv5 Repository**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  
- **Official YOLOv5 Docs**: [Docs](https://docs.ultralytics.com/)  

📩 **Feel free to contribute or open an issue!**  

---

### 📌 **Author**  
👤 **Neeraj S**  
🔗 [GitHub](https://github.com/NRJ900) | [LinkedIn](https://www.linkedin.com/in/mani-s-neeraj/)  

---
