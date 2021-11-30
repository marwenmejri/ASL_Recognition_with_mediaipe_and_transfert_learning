import pickle
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import tensorflow as tf


# with open('sign_classifier.pkl', 'rb') as f:
#     model = pickle.load(f)

model = tf.keras.models.load_model("sign_classifier.h5")

class_names = [" A ", " B ", " C ", " E ", " G "]

# mp_drawing = mp.solutions.drawing_utils # Drawing helpers
# mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
            success, image = cap.read()

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                  for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())

                        # Extract Pose landmarks
                        hand = hand_landmarks.landmark
                        hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
                        # Make Detections
                        X = pd.DataFrame([hand_row])
                        pred = model.predict(X)
                        # print(tf.reduce_max(pred))
                        # print(pred)
                        # print(type(pred))
                        # print(tf.argmax(pred.flatten()))

                        sign_alphabet = class_names[tf.argmax(pred.flatten())]
                        sign_alphabet_prob = tf.reduce_max(pred).numpy()
                        print(sign_alphabet, sign_alphabet_prob)
                        # print(num)

                        # Grab MIDDLE_FINGER_MCP coords
                        coords = tuple(np.multiply(
                            np.array(
                                (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y))
                            , [640, 480]).astype(int))

                        cv2.rectangle(image,
                                      (coords[0] - 100, coords[1] - 100),
                                      (coords[0] + 100, coords[1] + 100),
                                      (245, 117, 16), 3)
                        cv2.putText(image, sign_alphabet, coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Get status box
                        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                        # Display Class
                        cv2.putText(image, 'Alphabet'
                                    , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, sign_alphabet, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Display Probability
                        cv2.putText(image, 'PROB'
                                    , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(sign_alphabet_prob, 2))
                                    , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(27) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()



