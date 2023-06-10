# File containing methods for computing oxygen saturation
# and alpha function (used to "solve" some issues with the quality of an image)
import someMethods as sm

import sys
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

# Capture a frame every 0.33 seconds (30 frames per 10 seconds)
frame_interval = 0.33  
frame_counter = 0
skipFrame = 0
frameFps = []
start_time = time.time()

# Since we want to have 30 frames per 10 seconds, here we define a constant to use in order to stop
# the process as soon as we reach 30 frames depending on how long we want the video to be
videoDuration = 10
MAX_FRAMES = 3 * videoDuration

# List of fingertips images
fingerTips = []

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

        # Time to be shown on the frame
        localTime = time.localtime()
        time_str = time.strftime("%H:%M:%S", localTime)

        # Displaying FPS on the image
        cv2.putText(frame, "FPS: " + str(int(fps)), (4, 15), cv2.FONT_HERSHEY_PLAIN, 1, (80,175,76), 1)
        cv2.putText(frame, "Time: " + time_str, (4, 35), cv2.FONT_HERSHEY_PLAIN, 1, (34,87,255), 1)

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

            if elapsed_time >= frame_interval:

                if skipFrame == 5:

                    cropped_frame = tempFrame[y_min:y_max, x_min:x_max]
                    fingerTips.append(cropped_frame)
                    frameFps.append(fps)
                    frame_counter += 1
                    start_time = time.time()

                else:

                    skipFrame += 1
                    start_time = time.time()
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('MediaPipe Hands', rgbFrame)

        # In order to stop recording press 'q' or you can just wait that 30 frames are stored
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if frame_counter < 10:
    print("Not enough samples have been collected.\nStopping execution.")
    sys.exit()

# We need integer values for each frame's FPS
frameFps = [int(value) for value in frameFps]

print("\n=====================================================================================================================")
print("We have collected %d images of our fingertip!" % len(fingerTips))
print("The capturing process is completed!\nProceeding with the estimation of the values for Blood Pressure, Oxygen Saturation and Hearth Beat!")
print("=====================================================================================================================")

# Example of images obtained
sm.plotImagesTitles(fingerTips[0], fingerTips[1], fingerTips[2], 
                    "RGB Ex. 1","RGB Ex. 2","RGB Ex. 3")

print("First step: computation of the mean and standard deviation of each collected image.")

# The mean of each image is compute as the sum of the values of its respective RGB channels and then divide it by  3
# The variance, instead, is computed as the average of the squared difference between each pixel’s RGB value and the overall mean RGB value
fingerTipsMeans = []
fingerTipsVariances = []

# Lists for each channels
redChannels = []
blueChannels = []
greenChannels = []

for x in range(len(fingerTips)):

    r,g,b = cv2.split(fingerTips[x])

    redChannels.append(r.mean())
    greenChannels.append(g.mean())
    blueChannels.append(b.mean())
    
    # Mean computation
    channelsMean = (r+g+b)/3
    fingerTipsMeans.append(channelsMean)

    # Variance computation
    average_rgb = np.mean(channelsMean)
    variance = np.mean((channelsMean - average_rgb) ** 2)
    fingerTipsVariances.append(np.sqrt(variance))

# Plot the 3 RGB channels
sm.plotRGBchannels(redChannels, greenChannels, blueChannels, "RGB Channels Before Being Processed")

print("The mean and standard deviation of each image have been computed!")
print("=====================================================================================================================")
print("Next step: computation of the ROIs for each collected image.")

# Here we will store all the ROIs obtained after checking if "couples" of pixels satisfy a threshold as suggested in the paper
fingersROI = []
redROIs = []
greenROIs = []
blueROIs = []

for x in range(len(fingerTips)):

    threshold = 1.9 * fingerTipsVariances[x]

    # This is going to store the values of each pixel that satisfy the threshold for a given image
    singleFingerROI = []
    singleFingerROIred = []
    singleFingerROIgreen = []
    singleFingerROIblue = []

    for row in range(fingerTips[x].shape[0]):

        tmpRow = []
        redROIrow = []
        greenROIrow = []
        blueROIrow = []

        # We move by 2 columns everytime so that we check "couples" of pixels that have not been checked before
        for column in range(fingerTips[x].shape[1]-1):

            diff = np.mean(fingerTips[x][row,column] - fingerTips[x][row,column+1])

            if(diff > threshold):
                continue

            redROIrow.append(fingerTips[x][row, column, 0])
            redROIrow.append(fingerTips[x][row, column+1, 0])

            greenROIrow.append(fingerTips[x][row, column, 1])
            greenROIrow.append(fingerTips[x][row, column+1, 1])

            blueROIrow.append(fingerTips[x][row, column, 2])
            blueROIrow.append(fingerTips[x][row, column+1, 2])

            tmpRow.append(fingerTips[x][row, column])
            tmpRow.append(fingerTips[x][row, column+1])
            
            column += 1

        singleFingerROI.append(tmpRow)
        singleFingerROIred.append(redROIrow)
        singleFingerROIgreen.append(greenROIrow)
        singleFingerROIblue.append(blueROIrow)

    fingersROI.append(singleFingerROI)
    redROIs.append(singleFingerROIred)
    greenROIs.append(singleFingerROIgreen)
    blueROIs.append(singleFingerROIblue)

print("%d ROIs computed!"  % len(fingersROI))
print("=====================================================================================================================")
print("Now we are going to check the results by reconstructing the images and plotting some of them.")

# Here we reconstruct images in order to compute the rPPG signals
reconstructedImages = sm.reconstructImages(fingerTips, fingersROI)
sm.plotImagesTitles(reconstructedImages[0], reconstructedImages[1], reconstructedImages[2], 
                    "ROI 1", "ROI 2", "ROI 3")

print("Next step: computation of the mean of each ROI's channel.")

# Here we compute the means for each ROI's channels
meanRedROIs = sm.meanComputation(redROIs)
meanBlueROIs = sm.meanComputation(blueROIs)

# We named this list rawrPPGSignals since its values, by definition, should be the mean of each greenROI
# and thats exactly what we do with the method sm.meanComputation
rawrPPGSignals = sm.meanComputation(greenROIs)

print("""
    %d means for the red channel computed!
    %d means for the green channel computed (these are the raw rPPG signals)!
    %d means for the blue channel computed!"
""" % (len(meanRedROIs), len(rawrPPGSignals), len(meanBlueROIs)))

print("=====================================================================================================================")
print("Now we are ready to compute the Oxygen Saturation and the signals of both Hearth Rate and Breath Rate!")

# We compute the mean fps so that we can use this in the alpha function
# we might want to have a different value for each frame but as of now it causes some issues
meanFps = np.mean(frameFps)

# Hearth Signal and Breath Signal computation
heartRateSignal, breathSignal = sm.alpha_function(meanRedROIs, rawrPPGSignals, meanBlueROIs, meanFps)

# Plot the computed signals
sm.plotHearthBreathSignals(heartRateSignal, breathSignal)
print("=====================================================================================================================")
# Oxygen Saturation value computation
oxygenSaturation = sm.oxygen_saturation(meanBlueROIs, meanRedROIs)
print("Oxygen Saturation of the Individual (Sp02): %d%%" % oxygenSaturation)

# Hearth rate computation
heartRate = sm.high_peak(meanFps, len(heartRateSignal), heartRateSignal, 50, 180)
print("Hearth Rate of the Individual: %d bpm" % heartRate)

# Breath rate computation
breathRate = sm.high_peak(meanFps, len(breathSignal), breathSignal, 10, 40)
print("Breath Rate of the Individual: %d bpm" % breathRate)
print("=====================================================================================================================")