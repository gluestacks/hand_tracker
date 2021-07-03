import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
x_ = int(180)
y_ = int(110)
w_ = int(640-2*x_)
h_ = int(480-2*y_)
clr_main = (0, 255, 255)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
counter = 0
landmarks_x = []
landmarks_y = []
landmarks_list = ['mp_hands.HandLandmark.WRIST',
                  'mp_hands.HandLandmark.THUMB_CMC',
                  'mp_hands.HandLandmark.THUMB_MCP',
                  'mp_hands.HandLandmark.THUMB_IP',
                  'mp_hands.HandLandmark.INDEX_FINGER_MCP',
                  'mp_hands.HandLandmark.INDEX_FINGER_PIP',
                  'mp_hands.HandLandmark.INDEX_FINGER_DIP',
                  'mp_hands.HandLandmark.INDEX_FINGER_TIP',
                  'mp_hands.HandLandmark.MIDDLE_FINGER_MCP',
                  'mp_hands.HandLandmark.MIDDLE_FINGER_PIP',
                  'mp_hands.HandLandmark.MIDDLE_FINGER_DIP',
                  'mp_hands.HandLandmark.MIDDLE_FINGER_TIP',
                  'mp_hands.HandLandmark.RING_FINGER_MCP',
                  'mp_hands.HandLandmark.RING_FINGER_PIP',
                  'mp_hands.HandLandmark.RING_FINGER_DIP',
                  'mp_hands.HandLandmark.RING_FINGER_TIP',
                  'mp_hands.HandLandmark.PINKY_MCP',
                  'mp_hands.HandLandmark.PINKY_PIP',
                  'mp_hands.HandLandmark.PINKY_DIP',
                  'mp_hands.HandLandmark.PINKY_TIP']
with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    cv2.rectangle(image, (x_, y_), (x_+w_, y_+h_), clr_main, 2)
    t3 = time.perf_counter()
    image_height, image_width, _ = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

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
      for hand_landmarks in results.multi_hand_landmarks:
##        print('hand_landmarks:', hand_landmarks)
##        print(
##            f'Index middle finger tip coordinates: (',
##            f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
##            f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height})'
##        )
        i = 0
        while i < 21:
          #landmarks_name = landmarks_list[int(i)]
          landmarks_x.append(int(hand_landmarks.landmark[i].x * image_width))
          landmarks_y.append(int(hand_landmarks.landmark[i].y * image_height))
          i += 1
        x = min(landmarks_x)
        y = min(landmarks_y)
        w = max(landmarks_x) - x
        h = max(landmarks_y) - y
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,155,255), 2)
        landmarks_x = []
        landmarks_y = []
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    t4 = time.perf_counter()
    print(f'{counter}: {1/(t4-t3):.2f} Hz')
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
