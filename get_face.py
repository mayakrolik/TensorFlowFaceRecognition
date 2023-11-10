import cv2
from time import sleep

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Caera failed to open")
    quit()
else:
    print("Camera ready for pictures")

sleep (1)

for i in range(100):
    _, frame = cap.read()
    cv2.imwrite(f'images/Maya/img_{i}.jpg', frame)

print("Hit enter when your face is not in front of the camera")
input()

for i in range(100):
    _, frame = cap.read()
    cv2.imwrite(f'images/no_face/img_{i}.jpg', frame)