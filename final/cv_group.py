#!/usr/bin/env python3

import numpy as np
import cv2
import dlib
from imutils import face_utils

class FaceLandmarksFinder(object):
    def __init__(self):
        # Pre-trained model from dlib, will use this and 
        # then extract the landmarks we are interested in
        p = "../shape_predictor_68_face_landmarks.dat"
        self._face_detector = dlib.get_frontal_face_detector()
        self._lm_predictor = dlib.shape_predictor(p)
    
    # Find the landmarks for a single face 
    # (first face found)
    def find_landmarks(self, color_frame, depth_frame):
        lms_mul, img = self.find_landmarks_multi(color_frame, depth_frame, 1)
        lms = lms_mul[0] if len(lms_mul) > 0 else None
        return lms, img
    
    # Finds landmarks for multiple faces 
    # (all faces found until limit is reached)
    def find_landmarks_multi(self, color_frame, depth_frame, limit=0):
        lms_multi = []
        image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        image = image.astype(np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        
        # Convert image to gray scale and get all faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self._face_detector(gray, 0)
        
        # Get image width and height, should be 640*480
        img_h, img_w = image.shape[0:2]

        # For each face
        for (i, rect) in enumerate(rects):
            # Find landmarks
            shape = self._lm_predictor(gray, rect)
            # Convert to numpy array
            shape = face_utils.shape_to_np(shape)
            # Extract only the 22 landmarks we are interested in
            shape = self._get_new_shape(shape)

            # Draw a small circle around each landmark on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Apparently, dlib returns landmark coordinates outside 
            # the image sometimes, then we just set depth to 0, 
            # since we don't have depth values for such coordinates.
            depth_array = np.array([depth_image[x, y] 
                                    if x < img_w and y < img_h else 0 
                                    for x, y in shape])
            # Normalize depth array to 0-255 values
            depth_array -= np.min(depth_array[:])
            depth_array /= np.max(depth_array[:])
            depth_array = (depth_array * 255)
            # Reshape depth array to vector to merge with shape array
            depth_array = np.reshape(depth_array, (-1, 1))

            # Merge x,y values for each landmark with 
            # the depth value of each landmark
            merged = np.append(shape, depth_array, axis=1)
            lms_multi.append(merged)

            # Break if we've found the specified limit of faces
            if limit > 0 and i >= limit:
                break
    
        return lms_multi, image

    # Extracts the 22 landmarks we are interested in 
    # from the 68 landmarks that are detected by dlib
    def _get_new_shape(self, shape):
        dlib_to_our_landmarks = [17, 19, 21, 22, 24, 26, 36, 39, 42, 45,
                                 31, 33, 35, 48, 51, 54, 62, 66, 57, 8]
        # dlib's 68 landmarks only include a landmark for the middle of the 
        # nose saddle, but we want landmarks to the left and right of the 
        # nose saddle. We calculate those landmarks using the landmarks for 
        # the inner eye corners and the nose saddle middle.
        nose_x = int((shape[42][0] - shape[39][0]) / 4)
        new_shape = [shape[i] for i in dlib_to_our_landmarks]
        new_shape[10:10] = [[shape[27][0] - nose_x, shape[27][1]],
                            [shape[27][0] + nose_x, shape[27][1]]]
        return new_shape
