<p align="center">
  <img src="./fmds_banner.png" alt="FMDS - Face Mask Detection System Banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Mayank-iitj/FMDS/stargazers"><img src="https://img.shields.io/github/stars/Mayank-iitj/FMDS?style=for-the-badge&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/Mayank-iitj/FMDS/forks"><img src="https://img.shields.io/github/forks/Mayank-iitj/FMDS?style=for-the-badge&color=blue" alt="GitHub forks"></a>
  <a href="https://github.com/Mayank-iitj/FMDS/issues"><img src="https://img.shields.io/github/issues/Mayank-iitj/FMDS?style=for-the-badge&color=red" alt="GitHub issues"></a>
  <a href="https://github.com/Mayank-iitj/FMDS/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Mayank-iitj/FMDS?style=for-the-badge&color=green" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=for-the-badge" alt="Python 3.x">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge" alt="TensorFlow 2.x">
</p>

# 😷 FMDS: Face Mask Detection System

The **Face Mask Detection System (FMDS)** is a robust, real-time computer vision application designed to automatically detect whether a person is wearing a face mask in static images or live video streams. Built on the power of Deep Learning and Computer Vision concepts, FMDS is an essential tool for ensuring public safety and compliance in various environments.

This project is proudly developed and maintained by **MAYANK SHARMA**.

## ✨ Key Features

*   **High Accuracy:** Achieves a reported **98% accuracy** in face mask detection.
*   **Real-Time Performance:** Optimized for real-time video stream processing.
*   **Lightweight Model:** Utilizes the **MobileNetV2** architecture, making it computationally efficient and ideal for deployment on edge devices like **Raspberry Pi** and **Google Coral**.
*   **Dual Functionality:** Supports detection in both **static images** and **live video feeds**.
*   **Robust Dataset:** Trained on a comprehensive dataset of **4095 real images** (2165 with mask, 1930 without mask).

## 🚀 System Architecture (Infographic)

The FMDS operates through a streamlined pipeline, leveraging a pre-trained deep learning model for rapid inference.

<p align="center">
  <img src="./fmds_architecture.png" alt="FMDS System Architecture Diagram" width="80%">
</p>

**Workflow:**

1.  **Input:** Image or video frame is captured.
2.  **Face Detection:** A Caffe-based face detector is used to localize faces.
3.  **Preprocessing:** Detected faces are cropped and resized.
4.  **Inference:** The processed image is fed into the **MobileNetV2** model.
5.  **Output:** A bounding box is drawn around the face, colored **green** for "Mask" and **red** for "No Mask."

## 🛠️ Tech Stack

The system is built using industry-leading technologies for computer vision and deep learning.

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | Keras / TensorFlow | Core framework for building and training the MobileNetV2 model. |
| **Computer Vision** | OpenCV | Handles image/video processing, frame capture, and drawing bounding boxes. |
| **Model Architecture** | MobileNetV2 | Lightweight, high-performance Convolutional Neural Network (CNN) for efficient inference. |
| **Face Localization** | Caffe-based Face Detector | Fast and accurate pre-trained model for initial face detection. |
| **Web Application** | Streamlit | Used to create a simple, interactive web interface for demonstration (`app.py`). |

## ⚙️ Installation Guide

### Prerequisites

Ensure you have **Python 3.x** installed. All required dependencies are listed in `requirements.txt`.

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Mayank-iitj/FMDS.git
    cd FMDS
    ```

2.  **Create and Activate Virtual Environment**

    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    # Create virtual environment
    python3 -m venv venv
    
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    
    # Activate the environment (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip3 install -r requirements.txt
    ```

## 💡 Usage

### 1. Training the Model (Optional)

If you wish to retrain the model, use the following command:

```bash
python3 train_mask_detector.py --dataset dataset
```

### 2. Detecting in a Static Image

To run the detection on a single image:

```bash
python3 detect_mask_image.py --image images/pic1.jpeg
```

### 3. Real-Time Video Detection

To run the detection on a live video stream (requires a webcam):

```bash
python3 detect_mask_video.py
```

### 4. Running the Web Application

The project includes a simple web application built with Streamlit:

```bash
streamlit run app.py
```

## 🏆 Awards and Recognition

This project has received significant recognition in the academic and developer communities:

*   **Runners Up** position in the **Amdocs Innovation India ICE Project Fair**.
*   Selected in **Devscript Winter Of Code**.
*   Selected in **Script Winter Of Code**.
*   Selected in **Student Code-in**.

## 📰 Citations and Academic Impact

The work behind FMDS has been cited in several reputable academic publications, demonstrating its impact on the field of computer vision and public health technology:

1.  [https://osf.io/preprints/3gph4/]()
2.  [https://link.springer.com/chapter/10.1007/978-981-33-4673-4\_49]()
3.  [https://ieeexplore.ieee.org/abstract/document/9312083/]()
4.  [https://link.springer.com/chapter/10.1007/978-981-33-4673-4\_48]()
5.  [https://www.researchgate.net/profile/Akhyar\_Ahmed/publication/344173985\_Face\_Mask\_Detector/links/5f58c00ea6fdcc9879d8e6f7/Face-Mask-Detector.pdf]()

## 🤝 Contribution

We welcome contributions from the community!

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please refer to the `CONTRIBUTING.md` for detailed guidelines.

## 👤 Author and Maintainer

This project is developed, owned, and maintained by:

### **MAYANK SHARMA**

*   **GitHub:** [Mayank-iitj](https://github.com/Mayank-iitj)
## 📜 License



***

*A project by Mayank Sharma, built with ❤️ and Deep Learning.*
***
