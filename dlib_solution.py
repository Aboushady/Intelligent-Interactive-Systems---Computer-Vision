#!/usr/bin/env python3

from imutils import face_utils
import dlib
import cv2
 
def get_new_shape(shape):
    nose = int((shape[42][0] - shape[39][0]) / 4)
    nose1 = [shape[27][0]-nose, shape[27][1]]
    nose2 = [shape[27][0]+nose, shape[27][1]]
    new_shape = []
    new_shape.append(shape[17])
    new_shape.append(shape[19])
    new_shape.append(shape[21])
    new_shape.append(shape[22])
    new_shape.append(shape[24])
    new_shape.append(shape[26])
    new_shape.append(shape[36])
    new_shape.append(shape[39])
    new_shape.append(shape[42])
    new_shape.append(shape[45])
    new_shape.append(nose1)
    new_shape.append(nose2)
    new_shape.append(shape[31])
    new_shape.append(shape[33])
    new_shape.append(shape[35])
    new_shape.append(shape[48])
    new_shape.append(shape[51])
    new_shape.append(shape[54])
    new_shape.append(shape[62])
    new_shape.append(shape[66])
    new_shape.append(shape[57])
    new_shape.append(shape[ 8])
    return new_shape
 
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        shape = get_new_shape(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Show the image
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()