#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2

import cv_group
import ml_group
import es_group

def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)

    return config

def main():
    # Set up RealSense camera
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    
    # Initialize face landmark finder with dlib's 
    # pre-trained model for 68 landmarks (CV group)
    cv_lm_finder = cv_group.FaceLandmarksFinder()
    
    # Start capturing images
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = frame_aligner.process(frames)
        c_frame = aligned_frames.get_color_frame()
        d_frame = aligned_frames.get_depth_frame()
        
        # Get face landmarks for first face found (CV group)
        landmarks, image = cv_lm_finder.find_landmarks(c_frame, d_frame)
        
        # If a face was found, predict its emotions (ML group)
        # and present them using NAO and IrisTK (ES group)
        if landmarks is not None:
            emotions = ml_group.predict_emotions(landmarks)
            es_group.present_emotions(emotions)
        
        # This shows the image with the landmarks painted onto it
        # Maybe doesn't belong in final product
        cv2.imshow("Output", image)
        
        # Quit streaming if Escape is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
    # Terminates the window where images are printed
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
