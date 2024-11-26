import cv2
import pyrealsense2 as rs
import numpy as np

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_frame[y, x]
        print(f"HSV Value at ({x}, {y}): {hsv_value}")

# Create a window and set the mouse callback function
cv2.namedWindow("RealSense")
cv2.setMouseCallback("RealSense", get_hsv_value)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the BGR image to HSV
        hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Show the image
        cv2.imshow("RealSense", color_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()