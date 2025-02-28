# Facial Emotion Recognition
A real-time facial emotion recognition project that leverages transfer learning with MobileNetV2 to classify facial expressions from live webcam input. This project processes the FER2013 dataset to train a deep learning model and integrates face detection with Haar Cascades to run emotion recognition on a live video stream.
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Real-Time Webcam Integration](#real-time-webcam-integration)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project aims to recognize facial emotions from a live webcam feed by:
- Preprocessing the FER2013 dataset to ensure consistent, high-quality input images.
- Implementing transfer learning with a pre-trained MobileNetV2 model that’s fine-tuned to classify the seven emotions: **angry, disgust, fear, happy, neutral, sad,** and **surprised**.
- Extending functionality to a real-time demo that uses OpenCV for webcam capture and Haar Cascade for face detection.

## Project Structure
The repository is organized following the below scheme:
```sh
Facial-Emotion-Recognition/
├── data/
│   ├── FER2013/train 
|   ├── FER2013/test        
│   └── processed          
├── models/
│   └── finetunedModel.h5             
├── frontEnd/
│   └── app.py 
├── train/
│   ├── train.py       
│   └── utils.py         
├── requirements.txt         
├── README.md                
└── LICENSE                  
```
## Data Preprocessing
The FER2013 dataset contains grayscale images labeled with seven emotions. The preprocessing pipeline includes:
- **Conversion to RGB:** Converting grayscale images to RGB if needed.
- **Resizing:** Standardizing images (e.g., to 224×224) to match the input size expected by MobileNetV2.
- **Storage:** Saving processed images in the `data/processed/` directory for efficient loading during training.

These steps ensure that the model receives consistent and high-quality data.

## Model Training
This project employs transfer learning with MobileNetV2:
- **Base Model:** A MobileNetV2 pre-trained on ImageNet is loaded.
- **Customization:**  
  - The original final classification layer (typically with 1000 outputs) is removed.
  - New dense layers are added after the global average pooling layer to gradually reduce the output size to match the 7 emotion classes.
- **Activation Functions:** ReLU is used in the intermediate dense layers and softmax in the final layer to produce probability scores.
- **Compilation:** The model is compiled with:
  ```python
  new_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  ```
## Real-Time Webcam Integration
The project extends its functionality to run on live video:

- **Face Detection**: Uses Haar Cascade (via OpenCV) to detect faces in each frame from the webcam.
- **Emotion Prediction**: 
  - Extracted face regions are preprocessed and fed into the trained model. 
  - The emotion with the highest softmax score is selected.
- **Mapping**: A simple dictionary maps the predicted index to the corresponding emotion:
```python
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
```
- **Display**: The webcam script (src/webcam.py) overlays the predicted emotion on the live video feed.
## Installation
Clone the repository:
```bash
git clone https://github.com/hamzabhat/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage
### Training the Models
Run the training scripts for the model:
```bash
python train/train.py
```
### Webcam
Launch the webcam demo:

```bash
python frontEnd/app.py
```
## Contributing
Contributions are welcome! To contribute, follow these steps:  

1. **Fork the repository** on GitHub.  
2. **Clone your fork** locally:  
   ```bash
   git clone https://github.com/your-username/Facial-Emotion-Recognition.git
   ```
3. **Create a new branch** for your feature or fix:
```
git checkout -b feature-branch
```
4. **Make changes and commit them**:
```
git commit -m "Added a new feature"
```
5. **Push to your fork** and **create a pull request**:
```
git push origin feature-branch
```
4. **Submit a pull request** and describe your changes.

#### Feel free to open an issue for any improvements or questions!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for further details
