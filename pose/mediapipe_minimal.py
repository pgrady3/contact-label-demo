import cv2
import mediapipe as mp
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import os


class MediaPipeWrapper:
    def __init__(self, aggressive=False):
        # self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

        if not aggressive:
            self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2)
        else:
            self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.1)

    def run_img(self, image, vis=False):
        out_dict = dict()
        out_dict['right_points'] = None
        out_dict['right_confidence'] = 0
        out_dict['left_points'] = None
        out_dict['left_confidence'] = 0

        # image = cv2.flip(image, 1)    # by default, the handedness is wrong!!!
        image_height, image_width, _ = image.shape
        results = self.hands.process(image)
        # print(results.multi_handedness)

        if results.multi_hand_landmarks is not None:
            for i in range(len(results.multi_hand_landmarks)):
                hand_points = np.zeros((21, 3))
                landmarks = results.multi_hand_landmarks[i].landmark

                for j in range(21):
                    hand_points[j, 0] = landmarks[j].x
                    hand_points[j, 1] = landmarks[j].y
                    hand_points[j, 2] = landmarks[j].z

                hand_points = hand_points.tolist()

                detection = results.multi_handedness[i].classification[0]
                if detection.label == 'Left':     # mediapipe expects flipped images, so handedness is reversed
                    out_dict['right_points'] = hand_points
                    out_dict['right_confidence'] = detection.score
                else:
                    out_dict['left_points'] = hand_points
                    out_dict['left_confidence'] = detection.score

        if vis:
            annotated_image = image.copy()
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

            out_dict['vis'] = annotated_image

        return out_dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.hands.close()

