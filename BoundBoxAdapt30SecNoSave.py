# File containing methods for computing oxygen saturation
# and alpha function (used to "solve" some issues with the quality of an image)
import someMethods as sm

import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)  # Change the parameter to a video file path if needed

imgDir = "C:\\Users\\media\\Desktop\\Uni\\Magistrale\\PrimoAnno\\SecondoSemestre\\Computer Vision\\Project\\fingertip_rPPG\\images\\"
fingertipsDir = "C:\\Users\\media\\Desktop\\Uni\\Magistrale\\PrimoAnno\\SecondoSemestre\\Computer Vision\\Project\\fingertip_rPPG\\fingertips\\"

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

# Capture a frame every 0.33 seconds (30 frames per 10 seconds)
frame_interval = 0.33  
frame_counter = 0
frame_no = 0
start_time = time.time()

# Since we want to have 30 frames per 10 seconds, here we define a constant to use in order to stop
# the process as soon as we reach 30 frames depending on how long we want the video to be
videoDuration = 10
MAX_FRAMES = 3*videoDuration

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

        localTime = time.localtime()
        time_str = time.strftime("%H:%M:%S", localTime)

        # Displaying FPS on the image
        cv2.putText(frame, "FPS: " + str(int(fps)), (3, 10), cv2.FONT_HERSHEY_PLAIN, 1, (80,175,76), 1)
        cv2.putText(frame, "Time: " + time_str, (3, 30), cv2.FONT_HERSHEY_PLAIN, 1, (34,87,255), 1)

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

                # ROI = index fingertip
                cropped_frame = tempFrame[y_min:y_max, x_min:x_max]
                fingerTips.append(cropped_frame)

                # These two lines might be deleted since they were used to store images of the fingertips and the frame itself
                # cv2.imwrite(fingertipsDir + str(frame_counter) + ".png", cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                # cv2.imwrite(imgDir + str(frame_counter) + ".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_counter += 1
                start_time = time.time()
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('MediaPipe Hands', rgbFrame)

        # In order to stop recording press 'q' or you can just wait that 30 frames are stored
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("=====================================================================================================================")
print("We have collected %d images of our fingertip!" % len(fingerTips))
print("The capturing process is completed!\nProceeding with the estimation of the values for Blood Pressure, Oxygen Saturation and Hearth Beat!")
print("=====================================================================================================================")

# Example of images obtained
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(fingerTips[0])
ax[0].set_title("Example 1", fontsize = 10)
ax[1].imshow(fingerTips[1])
ax[1].set_title("Example 2", fontsize = 10)
ax[2].imshow(fingerTips[2])
ax[2].set_title("Example 3", fontsize = 10)
plt.show()

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
rx = range(len(redChannels))
gx = range(len(greenChannels))
bx = range(len(blueChannels))

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(12, 6))

ax[0].plot(rx, redChannels, color = 'red')
ax[0].set_ylim(min(redChannels)-10,max(redChannels)+10)
ax[0].grid(True)

ax[1].plot(gx, greenChannels, color = 'green')
ax[1].set_ylim(min(greenChannels)-10,max(greenChannels)+10)
ax[1].grid(True)

ax[2].plot(bx, blueChannels, color = 'blue')
ax[2].set_ylim(min(blueChannels)-10,max(blueChannels)+10)
ax[2].grid(True)
plt.show()

# Here we will store all the ROIs obtained after checking if "couples" of pixels 
# satisfy a threshold as suggested in the paper
fingersROI = []

for x in range(len(fingerTips)):

    threshold = 1.9 * fingerTipsVariances[x]

    # This is going to store the values of each pixel that satisfy the threshold for a given image
    singleFingerROI = []

    for row in range(fingerTips[x].shape[0]):
        tmpRow = []
        # We move by 2 columns everytime so that we check "couples" of pixels that have not been checked before
        for column in range(fingerTips[x].shape[1]-1):

            mean1 = np.mean(fingerTips[x][row,column])
            mean2 = np.mean(fingerTips[x][row,column+1])

            diff = np.abs(mean1 - mean2)

            if(diff > threshold):
                continue

            tmpRow.append(fingerTips[x][row,column])
            tmpRow.append(fingerTips[x][row,column+1])
            
            column += 1

        singleFingerROI.append(tmpRow)

    fingersROI.append(singleFingerROI)

print("%d ROIs computed!\nProceeding with the reconstruction of the images." % len(fingersROI))

# Here we reconstruct the images starting from the pixels we stored for each ROI and
# after that we try to plot some of them to check the results
reconstructedImages = []

for roi in fingersROI:
    height, width, channels = fingerTips[0].shape




    reconstructedImage = np.zeros((height, width, channels), dtype=np.uint8)

    for row_idx, row in enumerate(roi):
        for col_idx, (pixel1, pixel2) in enumerate(zip(row[::2], row[1::2])):
            
            print("row", str(row_idx))
            print("col", str(col_idx))

            if col_idx * 2 < width:
                reconstructedImage[row_idx, col_idx * 2] = pixel1
            if col_idx * 2 + 1 < width:
                reconstructedImage[row_idx, col_idx * 2 + 1] = pixel2

    reconstructedImages.append(reconstructedImage)

print("%d images reconstructed!" % len(reconstructedImages))

sm.plot_images(fingersROI[0], fingersROI[1], fingersROI[2])