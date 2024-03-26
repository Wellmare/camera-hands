import cv2
import mediapipe as mp
import time
import math

import numpy as np

from HandTrackingModule import *

def main():
    cap = cv2.VideoCapture("http://192.168.0.18:8080/videofeed")
    # cap = cv2.VideoCapture(0)

    detector = handDetector()
    height, width = 560, 720
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert the blank image to white
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    thumb_mask = np.zeros_like(canvas)

    prev_x, prev_y = None, None
    prev_cursor = None
    while True:
        success, img = cap.read()
        detector.findHands(img)
        lmList, bbox = detector.findPosition(img, handNo=0, draw=False)

        if (
            detector.results.multi_hand_landmarks
            and len(detector.results.multi_hand_landmarks) == 1
            and len(lmList) != 0
        ):
            fingersUp = detector.fingersUp()

            x, y = lmList[8][1], lmList[8][2]  # конец указательного пальца
            # cursor
            if prev_cursor:
                cv2.circle(thumb_mask, prev_cursor, 3, (0, 0, 0), -1)
            cv2.circle(thumb_mask, (x, y), 3, (255, 255, 255), -1)
            prev_cursor = (x, y)

            # указательный палец
            if fingersUp == [0, 1, 0, 0, 0] or fingersUp == [1, 1, 0, 0, 0]:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 14)

                cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(canvas, (x, y), 10, (0, 255, 0), -1)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = None, None

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", cv2.add( canvas, thumb_mask))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
