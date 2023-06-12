# Vital Estimation from Fingertip Video Images

This project aims to estimate the oxygen saturation and vital signs (heart rate and breath rate) of an individual using the rPPG (remote photoplethysmography) technique.
The project utilizes computer vision and image processing techniques to analyze fingertip images captured from a webcam.

## Prerequisites
Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV
- Mediapipe
- Matplotlib
- NumPy
- Scipy

You can install the required dependencies by running the following command:

Copy code
pip install opencv-python mediapipe matplotlib numpy

## Usage

1. Clone the project repository and navigate to the project directory.
2. Run the `Vital_Estimation_Fingertip.py` script using Python.

The script will access your webcam and start capturing fingertip images.
Press 'q' to stop the image capture process or wait until the needed frames are collected.
The mean and standard deviation of each collected image will be computed.
Regions of interest (ROIs) will be extracted from the fingertip images.
Reconstructed images will be displayed to visualize the ROIs.
The mean values for each ROI's channel will be computed.
The heart rate and breath rate signals will be calculated using the `alpha_function` method.
The computed signals will be plotted for visualization.
The oxygen saturation, heart rate, and breath rate of the individual will be estimated using the `high_peak` and `oxygen_saturation` methods and will then be displayed.
Note: Make sure your environment has proper lighting conditions and that your fingertip is clearly visible in the webcam feed.

## Output
The script will output the following information:

- Estimation of oxygen saturation;
- Estimation of heart rate;
- Estimation of breath rate.


## Plot examples

### 1. MediaPipe's Hand Landmarks
<img width="1073" alt="hand-landmarks-mediapipe" src="https://github.com/justivanr/fingertip_rPPG/assets/83827643/c8330501-2a2a-4ae8-be5e-86817fc920ce">

#### 1.1 How it works using our system
![handLandmarksExample](https://github.com/justivanr/fingertip_rPPG/assets/83827643/c68f79d1-3885-49e4-bf9e-25ee7f49006a)

### 2. RGB Channels before being Processed
![exampleRGBGraph](https://github.com/justivanr/fingertip_rPPG/assets/83827643/c5fc2b9e-e15c-45fe-86b0-e14ca480a2fd)

### 3. ROIs RGB Channels Means (after being Processed)
![exampleROIRGBMeans](https://github.com/justivanr/fingertip_rPPG/assets/83827643/c880e8ff-30ea-4447-8743-902401df7b37)

### 4. Heart and Breath Rate Signals
![exampleHRBRsignals](https://github.com/justivanr/fingertip_rPPG/assets/83827643/0fa1b8b5-7357-428b-981a-a265c42b7196)

