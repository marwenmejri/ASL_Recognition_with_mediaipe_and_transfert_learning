import cv2
import mediapipe as mp
import numpy as np
import csv

class_name = "B"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
cap = cv2.VideoCapture(0)
# cap.open("http://192.168.100.130:8080/video")
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8) as hands:
    while True:
        success, image = cap.read()
        # image = cv2.imread(r"C:\Users\USER\Desktop\Downloaded_images\D2.jpg")
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.resize(image, (500, 500))
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                              mp_drawing_styles.get_default_hand_landmarks_style(),
                              mp_drawing_styles.get_default_hand_connections_style())
                print(hand_landmarks.landmark)

                num_coords = len(hand_landmarks.landmark)
                landmarks = ['class']
                for val in range(1, num_coords + 1):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
                    # print(landmarks)
                    # Export coordinates
                try:
                    # Extract Pose landmarks
                    hand = hand_landmarks.landmark
                    hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
                    # Append class name
                    hand_row.insert(0, class_name)

                    # Export to CSV
                    # Create a csv file with only the columns name
                    # with open('coords.csv', mode='w', newline='') as f:
                    #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    #     csv_writer.writerow(landmarks)

                    # #Append rows to the csv file already created
                    with open('coords.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(hand_row)
                except:
                    pass

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) == ord("q"):
          break

cap.release()
cv2.destroyAllWindows()
















# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
#
# cap = cv2.VideoCapture(0)
# # cap.open("http://192.168.1.11:8080/video")
#
# # Iniate Holistic Model
# with mp_holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) as holistic:
#     while True:
#         success, frame = cap.read()
#         img_resized = cv2.resize(frame, (620, 480))
#         #Recolor Feed
#         img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
#         results = holistic.process(img)
#
#         #Recolor image back to BGR for Rendering
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#         # 2. Right hand
#         mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         # 3. Left Hand
#         mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         cv2.imshow("Frame", img)
#
#         if cv2.waitKey(1) == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()
