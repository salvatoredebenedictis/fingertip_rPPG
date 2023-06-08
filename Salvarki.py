import someMethods as sm
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import firwin
from sklearn import preprocessing
#-----------------------------------------------------------------------------------------------------------
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def oxygen_saturation(mean_signal_blue, mean_signal_red):
    
    # Standard Deviation of both red and blue channel's means
    std_blue = np.std(mean_signal_blue)
    std_red = np.std(mean_signal_red)

    # Spo2 - Beer/Lambert Law Formula
    rOr = (std_red/mean_signal_red)/(std_blue/mean_signal_blue)

    # Anemic Children
    spo2 = 101.6 - 5.834 * rOr
    meanSpo2 = np.mean(spo2)

    return (round(meanSpo2, 0))

# This function apply the chrome process on channels and return the heart and breath signals
def alpha_function_hr(red, green, blue, fps):
    
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    xS = 3*red-2*green
    yS = 1.5*red+green-1.5*blue

    fs = fps/0.33
    ntaps = 512
    # Frequency cut for the heart signal
    hr_lowcut = 0.6
    hr_highcut = 4.0

    #-------------------------------- HR -----------------------------------------------------------------
    hanning_hr_filter = bandpass_firwin(ntaps, hr_lowcut, hr_highcut, fs, window='hann')
    xF = scipy.signal.lfilter(hanning_hr_filter, 1, xS)
    yF = scipy.signal.lfilter(hanning_hr_filter, 1, yS)
    redF = scipy.signal.lfilter(hanning_hr_filter, 1, red)
    greenF = scipy.signal.lfilter(hanning_hr_filter, 1, green)
    blueF = scipy.signal.lfilter(hanning_hr_filter, 1, blue)
    alpha = np.std(xF)/np.std(yF)
    signal_hr_final = 3*(1-alpha/2)*redF-2*(1+alpha/2)*greenF+3*(alpha/2)*blueF
    return signal_hr_final

def alpha_function_br(red, green, blue, fps):
    
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    xS = 3*red-2*green
    yS = 1.5*red+green-1.5*blue

    fs = fps/0.33
    ntaps = 512

    #Frequency cut for the breathing signal
    br_lowcut = 0.17
    br_highcut = 0.67

    #--------------------------------------BR---------------------------------------------------------------
    hanning_br_filter = bandpass_firwin(ntaps, br_lowcut, br_highcut, fs, window='hanning')
    xF = scipy.signal.lfilter(hanning_br_filter, 1, xS)
    yF = scipy.signal.lfilter(hanning_br_filter, 1, yS)
    redF = scipy.signal.lfilter(hanning_br_filter, 1, red)
    greenF = scipy.signal.lfilter(hanning_br_filter, 1, green)
    blueF = scipy.signal.lfilter(hanning_br_filter, 1, blue)
    alpha = np.std(xF) / np.std(yF)
    # signal_final=xF-(alpha*yF) raggruppamento
    signal_br_final = 3 * (1 - alpha / 2) * redF - 2 * (1 + alpha / 2) * greenF + 3 * (alpha / 2) * blueF

    return signal_br_final

def high_peak(Fs,n,component,low,high):
    """Extract high peak to heart/breath rate """
    raw = np.fft.rfft(component)
    fft = np.abs(raw)
    freqs = float(Fs) / n * np.arange(n / 2 + 1)
    freqs = 60. * freqs
    idx = np.where((freqs > low) & (freqs < high))
    pruned = fft[idx]
    pfreq = freqs[idx]
    freqs = pfreq
    fft = pruned
    idx2 = np.argmax(fft)
    bpm = freqs[idx2]
    idx += (1,)
    return bpm

def plot_rois(rois_Green, rois_Blue, rois_Red):
    fig, ax = plt.subplots(nrows = 1, ncols = 3,figsize=(10, 5))

    ax[0].plot(range(len(rois_Green)), rois_Green, color='green')
    ax[0].set_ylim(min(rois_Green)-10, max(rois_Green)+10)
    ax[0].grid(True)
    ax[1].plot(range(len(rois_Red)), rois_Red, color='red')
    ax[1].set_ylim(min(rois_Red)-10, max(rois_Red)+10)
    ax[1].grid(True)
    ax[2].plot(range(len(rois_Blue)), rois_Blue, color='blue')
    ax[2].set_ylim(min(rois_Blue)-10, max(rois_Blue)+10)
    ax[2].grid(True)
    plt.show()

def roi_extraction(fingerTips, rois_Green, rois_Blue, rois_Red):
    for f in range(len(fingerTips)):
        avarage_RGB_values=[]
        for row in range(fingerTips[f].shape[0]):
                for col in range(fingerTips[f].shape[1]):
                    pixel = fingerTips[f][row,col,:]
                    mean_pixel = sum(pixel)/3
                    avarage_RGB_values.append(mean_pixel)

        avg=np.mean(avarage_RGB_values)
        variance = np.mean((avarage_RGB_values - avg)**2)
        std = np.sqrt(variance)

    #ROI channels
        roi_green=[]
        roi_blue=[]
        roi_red =[]

    #ROI Extraction if less than threshold 
        for row in range(fingerTips[f].shape[0]):
            for col in range(fingerTips[f].shape[1]-1):
                pixel = fingerTips[f][row,col,:]
                next_pixel = fingerTips[f][row,col+1,:]
                threshold = 1.9 * std
                if(np.mean(np.abs(pixel-next_pixel))<= threshold):
                    #Create ROI for Red
                    roi_red.append(pixel[0])
                    roi_red.append(next_pixel[0])
                    #Create ROI for Green
                    roi_green.append(pixel[1])
                    roi_green.append(next_pixel[1])
                    #Create ROI for Blue
                    roi_blue.append(pixel[2])
                    roi_blue.append(next_pixel[2])

        #append the mean of pixel for each channel
        rois_Green.append(np.mean(roi_green))
        rois_Blue.append(np.mean(roi_blue))
        rois_Red.append(np.mean(roi_red))

def main_capture(mp_drawing, mp_hands, cap, previousTime, frameFps, frame_interval, frame_counter, skipped_frame, start_time, MAX_FRAMES, fingerTips):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened() and frame_counter < MAX_FRAMES:
            ret, frame = cap.read()
            tempFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not ret:
                break

        # Calculating the FPS
            currentTime = time.time()
            fps = 1 / (currentTime-previousTime)
            previousTime = currentTime

            localTime = time.localtime()
            time_str = time.strftime("%H:%M:%S", localTime)

        # Displaying FPS on the image
            cv2.putText(frame, "FPS: " + str(int(fps)), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (80,175,76), 1)
            cv2.putText(frame, "Time: " + time_str, (5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (34,87,255), 1)

        # Convert the BGR image to RGB and process it with Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

        # Clear the frame and draw the hand landmarks
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
               
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordinates of the index finger tip and DIP landmarks
                index_finger_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_dip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                image_height, image_width, _ = frame.shape
                tip_x, tip_y = int(index_finger_tip_landmark.x * image_width), int(index_finger_tip_landmark.y * image_height)
                dip_x, dip_y = int(index_finger_dip_landmark.x * image_width), int(index_finger_dip_landmark.y * image_height)

            # Distance between index finger tip and DIP landmarks
                distance = ((tip_x - dip_x) ** 2 + (tip_y - dip_y) ** 2) ** 0.5

                bbox_size = int(distance/2)
                x_min, y_min = tip_x - bbox_size, tip_y - bbox_size
                x_max, y_max = tip_x + bbox_size, tip_y + bbox_size*2
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw a circle at the index finger tip landmark
                cv2.circle(frame, (tip_x, tip_y), 5, (255, 0, 0), -1)

                elapsed_time = time.time() - start_time
                if elapsed_time >= frame_interval and skipped_frame == 1:
                # ROI = index fingertip
                    cropped_frame = tempFrame[y_min:y_max, x_min:x_max]
                    fingerTips.append(cropped_frame)
                    frame_counter += 1
                    start_time = time.time()
                    frameFps.append(int(fps))
                else:   
                    skipped_frame = 1
        
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('MediaPipe Hands', rgbFrame)

        # In order to stop recording press 'q' or you can just wait that 30 frames are stored
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Model mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

# Capture a frame every 0.33 seconds (30 frames per 10 seconds)
frameFps = []
frame_interval = 0.33  
frame_counter = 0
skipped_frame=0
start_time = time.time()

# Since we want to have 30 frames per 10 seconds, here we define a constant to use in order to stop
# the process as soon as we reach 30 frames depending on how long we want the video to be
videoDuration = 30
MAX_FRAMES = 3*videoDuration

# List of fingertips images
fingerTips = []

#Main function which start the capture of video in order to collect index finger tip frames
main_capture(mp_drawing, mp_hands, cap, previousTime, frameFps, frame_interval, frame_counter, skipped_frame, start_time, MAX_FRAMES, fingerTips)

print("=====================================================================================================================")
print("We have collected %d images of our fingertip!" % len(fingerTips))
print("The capturing process is completed!\nProceeding with the estimation of the values for Blood Pressure, Oxygen Saturation and Hearth Beat!")
print("=====================================================================================================================")

# Plot of bounding box
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(fingerTips[0])
ax[0].set_title("Example 1", fontsize = 10)
ax[1].imshow(fingerTips[1])
ax[1].set_title("Example 2", fontsize = 10)
ax[2].imshow(fingerTips[2])
ax[2].set_title("Example 3", fontsize = 10)
plt.show()

#ROI Extraction--------------------------------------------------------------------------
rois_Green=[]
rois_Blue=[]
rois_Red =[]

roi_extraction(fingerTips, rois_Green, rois_Blue, rois_Red)
plot_rois(rois_Green, rois_Blue, rois_Red)

#Computation-----------------------------------------------------------------------------
hr=alpha_function_hr(rois_Red,rois_Green,rois_Blue,np.mean(frameFps))
br=alpha_function_br(rois_Red,rois_Green,rois_Blue,np.mean(frameFps)) 
spo2 = oxygen_saturation(rois_Blue,rois_Red)


#plot of hr signal
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(range(len(hr)), hr, color='red')
ax.set_ylim(min(hr), max(hr))
ax.grid(True)
plt.show()

#print the results
print("SpO2:",int(spo2),"%")
print("Heart rate:", int(high_peak(np.mean(frameFps),len(hr),hr,40,180)),"bpm")
print("Breath rate: ",int(high_peak(np.mean(frameFps),len(br),br,10,40)))

