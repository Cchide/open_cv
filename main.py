import cv2
import pandas as pd
import os


image_path = 'download (2).jpeg'  
cascade_path = 'haarcascade_frontalface_default.xml'  


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path
    if 'haarcascade' in cascade_path else cascade_path)

# Load image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


face_data = []

# Draw rectangles around the faces
for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # print(f"✅ Done! {len(faces)} face(s) detected and saved to detected_faces.csv")
# from turtledemo.nim import COLOR
# import cv2

# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# image = cv2.imread('toby.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #cv2.imshow("Gray", "gray")
# #cv2.waitKey()

# Faces = face_haar_cascade.detectMultiScale(gray, 1.1 , 4)

# for (x, y, w, h) in Faces:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

# cv2.imshow("Faces", image)
# cv2.waitKey()
