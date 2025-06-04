# Traffic-Sign-Recognition-System

A real-time traffic sign detection and classification system built using deep learning and computer vision techniques. This project can identify various traffic signs from images or webcam feed using a Convolutional Neural Network (CNN) model.

## Features

- Real-time traffic sign detection and classification
- Support for both image upload and webcam input
- Enhanced image preprocessing for better accuracy
- Confidence score display for predictions
- Top 5 predictions visualization
- User-friendly Streamlit interface
- Support for 43 different traffic sign classes

## Technologies Used

- Python 3.x
- TensorFlow/Keras
- OpenCV
- Streamlit
- NumPy
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rayyan-mohammed/traffic-sign-recognition.git
cd traffic-sign-recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app_new.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Choose your input method:
   - Upload an image containing traffic signs
   - Use your webcam to capture traffic signs in real-time

4. View the detection results and confidence scores

## Model Details

- The system uses a pre-trained CNN model (model.h5)
- Input image size: 32x32 pixels
- Confidence threshold: 75%
- Enhanced preprocessing pipeline including:
  - Grayscale conversion
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Normalization

## Dataset

The model is trained on a comprehensive traffic sign dataset containing 43 different classes of traffic signs, including:
- Speed limits
- Warning signs
- Mandatory signs
- Priority signs
- And more

## Tips for Better Detection

1. Ensure the sign is clearly visible and centered
2. Capture in good lighting conditions
3. Avoid extreme angles or glare
4. Crop unnecessary background if possible
5. Try multiple angles if detection fails

## Project Structure

```
traffic-sign-recognition/
├── app_new.py              # Main Streamlit application
├── model.h5               # Pre-trained CNN model
├── labels.csv             # Traffic sign class labels
└──Dataset/              # Training dataset
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset
- Built with Streamlit for the user interface
- Powered by TensorFlow/Keras for deep learning capabilities 
