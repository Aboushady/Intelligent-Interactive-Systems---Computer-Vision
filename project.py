#!/usr/bin/env python3

from imutils import face_utils
import pyrealsense2 as rs
import numpy as np
import dlib
import cv2

file_index = 0


def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)

    return config


def get_new_shape(shape):
    dlib_to_our_landmarks = [17, 19, 21, 22, 24, 26, 36, 39, 42, 45,
                             31, 33, 35, 48, 51, 54, 62, 66, 57, 8]
    nose_x = int((shape[42][0] - shape[39][0]) / 4)
    new_shape = [shape[i] for i in dlib_to_our_landmarks]
    new_shape[10:10] = [[shape[27][0] - nose_x, shape[27][1]],
                        [shape[27][0] + nose_x, shape[27][1]]]
    return new_shape


def main(callback):
    # Pre-trained model from dlib, will use this and 
    # then extract the landmarks we are interested in
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Set up RealSense camera
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    # clipping_distance_meters = 0.2
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # clipping_distance = clipping_distance_meters / depth_scale

    # Use web camera
    # cap = cv2.VideoCapture(0)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = frame_aligner.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        image = image.astype(np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # Use web camera
        # Getting out image by webcam 
        # _, image = cap.read()

        # Convert image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get all faces
        rects = detector(gray, 0)

        # For each face
        for (i, rect) in enumerate(rects):
            # Find landmarks
            shape = predictor(gray, rect)
            # Convert to numpy array
            shape = face_utils.shape_to_np(shape)
            # Extract only the 22 landmarks we are interested in
            shape = get_new_shape(shape)
            # Array to hold depth value for each landmark pixel
            depth_array = np.zeros(len(shape))

            # For each landmark
            for i, (x, y) in enumerate(shape):
                # Draw small circle around it on the image
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                # Save the landmark's depth information
                depth_array[i] = depth_image[x, y]

            # Normalize depth array to 0-255 values
            depth_array -= np.min(depth_array[:])
            depth_array /= np.max(depth_array[:])
            depth_array = (depth_array * 255)
            depth_array = np.reshape(depth_array, (-1, 1))

            # Merge x,y values for each landmark with 
            # the depth value of each landmark
            merged = np.append(shape, depth_array, axis=1)

            # Send the landmark information to the 
            # specified callback function
            # callback(merged)

            # Export landmarks to .lm3
            export(merged)

        # Show the image
        cv2.imshow("Output", image)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def export(data):
    file_name = file_index + ".lm3"
    file = open(file_name, "w")

    for entry in data:
        output = " ".join([str(v) for v in entry])
        file.write(output)

    file.close()
    file_index += 1


if __name__ == "__main__":
    main(test)
