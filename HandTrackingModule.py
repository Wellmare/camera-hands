import cv2
import mediapipe as mp
import time
import math

import numpy as np


class handDetector:
    def __init__(
        self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.1, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                    2,
                )
        return self.lmList, bbox

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    cap = cv2.VideoCapture("http://192.168.0.18:8080/videofeed")

    # Add a line to resize the frames to a smaller size (e.g., 480x360)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

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
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 3)

                cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(canvas, (x, y), 10, (0, 255, 0), -1)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = None, None

            # cv2.circle(img, (lmList[8][1] + 100, lmList[8][2]), 10, (0, 0, 255), -1)
        # большой палец
        # if len(lmList) != 0:
        #     x, y = lmList[4][1], lmList[4][2]
        #     cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", cv2.add( canvas, thumb_mask))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
