#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
from imutils import face_utils
import math
import time

lm_labels = [
    "Outer left eyebrow", 
    "Middle left eyebrow", 
    "Inner left eyebrow", 
    "Inner right eyebrow", 
    "Middle right eyebrow", 
    "Outer right eyebrow", 
    "Outer left eye corner", 
    "Inner left eye corner", 
    "Inner right eye corner", 
    "Outer right eye corner", 
    "Nose saddle left", 
    "Nose saddle right", 
    "Left nose peak", 
    "Nose tip", 
    "Right nose peak", 
    "Left mouth corner", 
    "Upper lip outer middle", 
    "Right mouth corner", 
    "Upper lip inner middle", 
    "Lower lip inner middle", 
    "Lower lip outer middle", 
    "Chin middle"
]

def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)
    return config

def find_c_and_d_lms(c_image, d_image, face_detector, lm_predictor):
    faces = []
    c_gray = c_to_gray(c_image)
    d_gray = d_to_gray(d_image)
    
    # Detect faces using RGB image
    rects = face_detector(c_gray, 0)
    
    for rect in rects:
        c_lms = predict_face(c_gray, rect, lm_predictor)
        d_lms = predict_face(d_gray, rect, lm_predictor)
        faces.append((c_lms, d_lms))

    return faces

def c_to_gray(c_image):
    return cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)

def d_to_gray(d_image):
    minn = np.min(d_image[d_image > 0])
    maxx = minn + 100
    d_image -= minn
    d_image /= maxx
    d_image = (d_image * 255)
    d_image = d_image.astype(np.uint8)
    return d_image
    
def predict_face(gray, rect, predictor):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    shape = get_new_shape(shape)
    return shape

# Extracts the 22 landmarks we are interested in 
# from the 68 landmarks that are detected by dlib
def get_new_shape(shape):
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

def draw_lms(image, lms, color):
    for (x, y) in lms:
        cv2.circle(image, (x, y), 2, color, -1)

def main():
    # Pre-trained model from dlib, will use this and 
    # then extract the landmarks we are interested in
    p = "../shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    lm_predictor = dlib.shape_predictor(p)
    
    # Set up RealSense camera
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    
    tot_distances = []
    frame_count = 0
    time_start = time.time()
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = frame_aligner.process(frames)
        c_frame = aligned_frames.get_color_frame()
        d_frame = aligned_frames.get_depth_frame()
        
        c_image = np.asanyarray(c_frame.get_data()).astype(np.float32)
        c_image = c_image.astype(np.uint8)
        d_image = np.asanyarray(d_frame.get_data()).astype(np.float32)
        
        faces = find_c_and_d_lms(c_image, d_image, face_detector, lm_predictor)
        
        
        for c_lms, d_lms in faces:
            distances = []
            for i in range(len(c_lms)):
                c_dist, d_dist = np.subtract(c_lms[i], d_lms[i])
                dist = math.sqrt(c_dist*c_dist + d_dist*d_dist)
                distances.append(dist)
            tot_distances.append(distances)
        
        for c_lms, d_lms in faces:
            draw_lms(c_image, c_lms, (0, 255, 0))
            draw_lms(c_image, d_lms, (0, 0, 255))
        
        cv2.imshow("C_IMAGE", c_image)
        #cv2.imshow("D_IMAGE", d_image)
        
        frame_count += 1
        
        # Quit streaming if Escape is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    
    time_end = time.time()
    
    each_mean = np.mean(np.array(tot_distances), axis=0)
    tot_mean = np.mean(each_mean)
    
    f = open("lms.txt", "w")
    
    f.write("Frame count: {}\n".format(frame_count))
    f.write("Time elapsed: {}\n".format(time_end - time_start))
    
    f.write("Mean of each landmark:\n\n".format(frame_count))
    
    for i in range(22):
        f.write("{0: <25}{1}\n".format(lm_labels[i], each_mean[i]))
    
    f.write("\n")
    f.write("Total mean of the means of each landmark: {}".format(tot_mean))
    f.close()
    
    print(each_mean)
    print(tot_mean)



if __name__ == "__main__":
    main()