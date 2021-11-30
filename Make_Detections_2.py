import pickle
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt

# with open('sign_classifier.pkl', 'rb') as f:
#     model = pickle.load(f)

# model = tf.keras.models.load_model("sign_classifier.h5")
model = tf.keras.models.load_model("SL_Model_Efficientnet.h5")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# mp_drawing = mp.solutions.drawing_utils # Drawing helpers
# mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def make_prediction(model, img, input_shape, class_names):

    image = cv2.resize(img, input_shape)
    img_reshaped = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    pred = model.predict(img_reshaped)

    pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    pred_confidence = np.max(pred)
    return pred_class, pred_confidence


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
                        pred_class, pred_confidence = make_prediction(model=model, img=image, input_shape=(224, 224), class_names=class_names)
                        print(pred_class, pred_confidence)
                        print(str(np.round(pred_confidence, 2)), " %")
                        prob = str(np.round(pred_confidence, 2)) + " %"

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
                        # cv2.putText(image, pred_confidence, coords,
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Get status box
                        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                        # Display Class
                        cv2.putText(image, 'Class: '
                                    , (125, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, pred_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

                        # Display Probability
                        cv2.putText(image, 'Confidence: '
                                    , (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, prob
                                    , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(27) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()



