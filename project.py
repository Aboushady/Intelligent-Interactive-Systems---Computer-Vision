#!/usr/bin/env python3

from imutils import face_utils
import pyrealsense2 as rs
import numpy as np
import dlib
import cv2
 
def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)

    return config

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


def main(callback):
    # let's go code an faces detector(HOG) and after detect the 
    # landmarks on this detected face

    # p = our pre-treined model directory, on my case, it's on the same script's diretory.
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # RealSense version
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    clipping_distance_meters = 0.2
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = clipping_distance_meters / depth_scale

    # Other version
    #cap = cv2.VideoCapture(0)

    while True:
        # RealSense version
        frames = pipeline.wait_for_frames()
        aligned_frames = frame_aligner.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        image = image.astype(np.uint8)
        
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        
        # Other version
        # Getting out image by webcam 
        #_, image = cap.read()
        
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
            
            depth_array = np.zeros(len(shape))
        
            # Draw on our image, all the finded cordinate points (x,y) 
            for i, (x, y) in enumerate(shape):
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                depth_array[i] = depth_image[x,y]
            
            depth_array -= np.min(depth_array[:])
            depth_array /= np.max(depth_array[:])
            depth_array = (depth_array * 255).astype(np.uint8)
            depth_array = np.reshape(depth_array, (-1, 1))
            
            merged = np.append(shape, depth_array, axis=1)
            
            callback(merged)
        
        # Show the image
        cv2.imshow("Output", image)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def test(data):
    print(data)
    
if __name__ == "__main__":
    main(test)
    