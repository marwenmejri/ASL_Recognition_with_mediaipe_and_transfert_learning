import pickle
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp


with open('sign_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

class_names = [" B ", " C ", " E "]

# mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
            success, image = cap.read()

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 2. Right hand
            img_right = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            mp_drawing.draw_landmarks(img_right, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            img_left = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            mp_drawing.draw_landmarks(img_left, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            if results.left_hand_landmarks:
                hand = results.left_hand_landmarks.landmark
                # print(hand)
                print(type(hand))
                for item in hand:
                    print(item.landmark)
                # hand_row = list(np.array([[hand["x"], hand["y"], hand["z"]] for landmark in hand]).flatten())
                # # Make Detections
                # X = pd.DataFrame([hand_row])
                # sign_alphabet = class_names[model.predict(X)[0]]
                # sign_alphabet_prob = model.predict_proba(X)[0]
                # print(f"Left Hand  : {sign_alphabet}, {sign_alphabet_prob}")
            if results.right_hand_landmarks:
                hand = results.right_hand_landmarks.landmark
                # print(hand)
                # hand_row = list(np.array([[hand.x, hand.y, hand.z] for landmark in hand]).flatten())
                # # Make Detections
                # X = pd.DataFrame([hand_row])
                # sign_alphabet = class_names[model.predict(X)[0]]
                # sign_alphabet_prob = model.predict_proba(X)[0]
                # print(f"right hand : {sign_alphabet}, {sign_alphabet_prob}")


            # # Grab MIDDLE_FINGER_MCP coords
            # coords = tuple(np.multiply(
            #     np.array(
            #         (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            #          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y))
            #     , [640, 480]).astype(int))
            #
            # cv2.rectangle(image,
            #               (coords[0] - 100, coords[1] - 100),
            #               (coords[0] + 100, coords[1] + 100),
            #               (245, 117, 16), 3)
            # # cv2.putText(image, sign_alphabet, coords,
            # #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #
            # # Get status box
            # cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            #
            # # Display Class
            # cv2.putText(image, 'Alphabet'
            #             , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, sign_alphabet.split(' ')[0]
            #             , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #
            # # Display Probability
            # cv2.putText(image, 'PROB'
            #             , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(round(sign_alphabet_prob[np.argmax(sign_alphabet_prob)], 2))
            #             , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', img_left)
            cv2.imshow('MediaPipe Left Hands', img_right)


            if cv2.waitKey(27) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()



