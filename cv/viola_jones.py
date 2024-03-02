""" This script is for using Viola-Jones method 
for face detection on webcam. 

The Paper: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

'Used in real-time applications, the detector runs at 15 
frames per second without resorting to image differencing 
or skin color detection.'

This was the first real application of CV, implemented in a 
digital camera as a face detector in 2006 by Fujifilm.
"""

import cv2
import os
from pathlib import Path

SRC_DIR = os.path.dirname(__file__)
ROOT = str(Path(SRC_DIR).parents[0])
BLOBS = os.path.join(ROOT, "blobs" )

# download the xmls from opencv repository
if not os.path.isfile(os.path.join(f"{BLOBS}", "haarcascade_frontalface_default.xml")):
    os.system(f"wget -P {BLOBS} https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")

detector = cv2.CascadeClassifier(os.path.join(f"{BLOBS}", "haarcascade_frontalface_default.xml"))

webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read() # True, Array

    # Viola Jones works on grey scale images:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rectangle = detector.detectMultiScale(
	                    gray, scaleFactor=1.05, minNeighbors=5, 
                        minSize=(30, 30),
	                    flags=cv2.CASCADE_SCALE_IMAGE)

    # if there is a bounding box detected.
    if len(face_rectangle) != 0:
        # get a single Bounding Box and draw on frame
        x, y, w, h = face_rectangle[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), 
                      (0, 255, 0), 2)

    cv2.imshow("Viola Jones Face ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        