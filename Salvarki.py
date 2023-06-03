import someMethods as sm
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

#Model mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
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

rPPG=[]
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

    roi=[]
    for row in range(fingerTips[f].shape[0]):
        for col in range(fingerTips[f].shape[1]-1):
            pixel = fingerTips[f][row,col,:]
            next_pixel = fingerTips[f][row,col+1,:]
            if(np.mean(np.abs(pixel-next_pixel))<= 1.9 * std):
                roi.append(pixel[1])
                roi.append(next_pixel[1])
    mean_green = np.mean(roi)
    rPPG.append(mean_green)


# Plot the RPPG
gx = range(len(rPPG))

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(gx, rPPG, color='green')
ax.set_ylim(min(rPPG)-10, max(rPPG)+10)
ax.grid(True)
plt.show()

"""
##ROI EXTRACTION
    #FingerTips[i] è una immagine (fingertips ne contiene 30). La sua forma è [riga,colonna,canalecolore]
    #pixel0 è [0,0,:] restituisce un array con i tre colori RGB dunque il risultato sarà pixel=[r,g,b]

#calcolare il colore medio di un pixel che è dato da R+G+B/3

pixels_medio=[] #array che contiene il valore medio di ogni pixel
frame_color_medio = [] # array che contiene i valori medi di ogni pixel per ogni immagine

#per ogni frame
for f in range(len(fingerTips)):
    pixels_medio=[] 
    #per ogni pixel (il pixel viene identificato da riga e colonna)
    for row in range(fingerTips[f].shape[0]):
        for col in range(fingerTips[f].shape[1]):
            pixel = fingerTips[f][row,col,:]
            mean_pixel = sum(pixel)/3
            pixels_medio.append(mean_pixel)
    frame_color_medio.append(pixels_medio)

variance = np.mean(pixel[0]- mean_pixel)
"""