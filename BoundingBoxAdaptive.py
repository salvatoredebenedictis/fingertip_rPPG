import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)  # Change the parameter to a video file path if needed

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

        # Convert the BGR image to RGB and process it with Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Clear the frame and draw the hand landmarks
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip and DIP landmarks
            index_finger_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_dip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            image_height, image_width, _ = frame.shape
            tip_x, tip_y = int(index_finger_tip_landmark.x * image_width), int(index_finger_tip_landmark.y * image_height)
            dip_x, dip_y = int(index_finger_dip_landmark.x * image_width), int(index_finger_dip_landmark.y * image_height)

            # Calculate the distance between index finger tip and DIP landmarks
            distance = ((tip_x - dip_x) ** 2 + (tip_y - dip_y) ** 2) ** 0.5

            # Adjust the size of the bounding box based on the distance
            bbox_size = int(distance/2)  # Adjust the divisor as needed
            x_min, y_min = tip_x - bbox_size, tip_y - bbox_size
            x_max, y_max = tip_x + bbox_size, tip_y + bbox_size*2
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw a circle at the index finger tip landmark
            cv2.circle(frame, (tip_x, tip_y), 5, (255, 0, 0), -1)

        cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
            break

cap.release()
cv2.destroyAllWindows()
