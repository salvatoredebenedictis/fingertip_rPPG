import cv2
import mediapipe as mp
import time

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

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened() and frame_counter < MAX_FRAMES:
        ret, frame = cap.read()
     
        if not ret:
            break

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime

        localTime = time.localtime()
        time_str = time.strftime("%M:%S", localTime)

        # Displaying FPS on the image
        cv2.putText(frame, str(int(fps))+" FPS: ", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, " Time: " + time_str, (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

        # Convert the BGR image to RGB and process it with Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Clear the frame and draw the hand landmarks
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        tempFrame = frame.copy()
        
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
                cv2.imwrite(fingertipsDir + str(frame_counter) + ".png", cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

                cv2.imwrite(imgDir + str(frame_counter) + ".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_counter += 1
                start_time = time.time()
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('MediaPipe Hands', rgbFrame)

        # In order to stop recording: press 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

print("=====================================================================================================================")
print("Process Completed!\nProceeding with the estimation of the values for Blood Pressure, Oxygen Saturation and Hearth Beat!")
print("=====================================================================================================================")

cv2.destroyAllWindows()
