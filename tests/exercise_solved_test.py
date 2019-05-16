import pyrealsense2 as rs
import numpy as np
import cv2


def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)

    return config


def create_segmentation_mask(x, y, w, h, image, depth_image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    average_depth = np.mean(depth_image[y:y+h, x:x+w])
    high = min(average_depth + clipping_distance, 2**16)
    low = max(average_depth - clipping_distance, 0)
    mask[np.logical_and(low <= depth_image, depth_image <= high)] = 1.0
    return mask

def print_to_file(arr):
    f = open("depth.txt", "w")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            f.write("{0: >5}".format(arr[r,c]))
        f.write("\n")
    f.close()
    
if __name__ == "__main__":
    m = None
    depth_image = None
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    background = np.zeros((480, 640, 3))

    # pretrained haar-cascade classifier data
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # configure the realsense camera
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    clipping_distance_meters = 0.2

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = clipping_distance_meters / depth_scale

    try:
        while cv2.getWindowProperty('RealSense', 0) != -1:
            # get a raw image into the pipeline
            frames = pipeline.wait_for_frames()
            aligned_frames = frame_aligner.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Preprocessing
            # image to display (normalized float bgr)
            image = np.asanyarray(color_frame.get_data()).astype(np.float32)
            image -= np.min(image[:])
            image /= np.max(image[:])

            # convert to grayscale (uint8)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image -= np.min(gray_image[:])
            gray_image /= np.max(gray_image[:])
            gray_image_uint = gray_image * 255
            gray_image_uint = gray_image_uint.astype(np.uint8)

            
            depth_image = np.asanyarray(depth_frame.get_data())
            #np.savetxt("depth.txt", depth_image, delimiter=" ", fmt="%1.2e")
            #print_to_file(depth_image)
            #break
            
            # Processing
            faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)
            if len(faces) > 0:
                mask = np.zeros((image.shape[0], image.shape[1]))
                print(depth_image.shape)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    mask = np.zeros((image.shape[0], image.shape[1]))
                    average_depth = np.mean(depth_image[y:y+h, x:x+w])
                    high = min(average_depth + clipping_distance, 2**16)
                    low = max(average_depth - clipping_distance, 0)
                    mask[np.logical_and(low <= depth_image, depth_image <= high)] = 1.0
                    
                    dm = depth_image[y:y+h, x:x+w]
                    dm[dm == 0] = 10000
                    #np.savetxt("test.txt", dm, delimiter=" ", fmt="%1.2e")
                    
                    m = np.unravel_index(dm.argmin(), dm.shape)
                    #print(m)
                    m = (m[1] + x, m[0] + y)
                    #print(m)
                    #print()
                    
                    #person_mask = create_segmentation_mask(x, y, w, h, image,
                    #                                       depth_image)
                    person_mask = mask
                    mask = np.logical_or(mask, person_mask)
            else:
                mask = np.ones((image.shape[0], image.shape[1]))

            
                
            # Postprocessing
            #image = (mask[..., None] * image + (1 - mask[..., None]) *
            #         background)
            image = (image * 255).astype(np.uint8)
            
            cv2.rectangle(image, (318, 238), (322, 242), (0, 255, 0), 2)
            
            if m is not None:
                cv2.rectangle(image, (m[0], m[1]), (m[0]+5, m[1]+5), (255, 0, 0), 2)
            

            #if depth_image is not None:
            #    image = depth_image
            
            # Display the final image
            cv2.imshow('RealSense', image)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break
    finally:
        # Stop streaming
        pipeline.stop()
