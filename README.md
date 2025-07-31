# 🚗 Implementation of a Surround View Camera System on NVIDIA Jetson

This project implements a real-time **surround-view image stitching system** using six **Arducam cameras** connected to an **NVIDIA Jetson AGX Orion** board. The system captures synchronized images from all cameras and stitches them into a single panoramic view, enabling enhanced spatial perception for autonomous systems.

---

## 📷 Functional Block Diagram  
<img width="1796" height="477" alt="image" src="https://github.com/user-attachments/assets/5bd8dfc1-87db-4ed5-9783-aa74da39d8c8" />


---

## 🚀 Features

- 📸 High-resolution image capture using **GStreamer** and **Arducam** drivers on Jetson  
- 🧵 Image stitching via **OpenCV**, with fallback to **AKAZE + Homography** when needed  
- 💾 Panorama is saved to disk and optionally previewed in real-time  
- 🔄 Supports **6-camera surround configuration** for wide-angle perception  

---

## 🛠️ Setup

### 🔌 Hardware
- NVIDIA Jetson AGX Orion  
- 6x Arducam CSI Cameras  

### 🧪 Software Dependencies

Install required Python packages:

```bash
pip3 install -r requirements.txt
