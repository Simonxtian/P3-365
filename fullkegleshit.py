import math
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

startTime = time.time()

halfImageWidth = 640 / 2
FOV = 87.0
width = 640
focalLength = width / (2 * math.tan(math.radians(FOV / 2)))

#grænseværdier for gul farve i HSV
nedreGul = (20,130,125)
ovreGul = (55,255,255)

#grænseværdier for blå farve i HSV
nedreBlaa = (100,200,0)
ovreBlaa = (120,255,255)

def get_cartesian_coordinates(x, y, w, depth_image, img, focal_length, half_image_width):
    """
    Converts pixel coordinates (x, y) of an object in an image to Cartesian coordinates
    relative to the camera, using the depth frame to calculate distance and focal length
    to calculate the angle.

    Parameters:
    - x, y: Pixel coordinates of the object.
    - depth_frame: The depth frame from which to retrieve the object's distance.
    - focal_length: The focal length of the camera in pixels.
    - image_width: The width of the image in pixels.

    Returns:
    - (x_cart, y_cart): Tuple of Cartesian coordinates in millimeters (integers).
    """
    
    
    # Calculate horizontal angle of the object from the center of the image
    dx = x - half_image_width
    angle = math.degrees(math.atan(dx / focal_length))

    # Get distance from the depth frame
    bbox = (x-w//5, y-w*2//5, w*2//5, w*2//5)
    # Draw bbox
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
    # average distance in the bounding box
    distance = np.mean(depth_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
    # print(distance)

    # Convert spherical coordinates to Cartesian
    x_cart = int(distance * math.cos(math.radians(angle)))
    y_cart = int(distance * math.sin(math.radians(angle)))

    return x_cart, y_cart

def calculate_midpoints(blue_cones, yellow_cones):
    """
    Calculates the midpoints between blue and yellow cones using Delaunay triangulation.

    Parameters:
    - blue_cones (array-like): List of (x, y) tuples representing blue cone coordinates.
    - yellow_cones (array-like): List of (x, y) tuples representing yellow cone coordinates.

    Returns:
    - midpoints (list of tuples): Midpoints between paired blue and yellow cones.
    - delaunay (Delaunay object): The Delaunay triangulation object for plotting triangles.
    """
    # Combine blue and yellow cones into a single array for triangulation
    all_cones = np.array(blue_cones + yellow_cones)
    if len(all_cones) < 4:
        midpoints = []
        for blue_cone in blue_cones:
            for yellow_cone in yellow_cones:
                midpoints.append(((blue_cone[0] + yellow_cone[0]) / 2, (blue_cone[1] + yellow_cone[1]) / 2))
        return midpoints
    else:
        num_blue = len(blue_cones)  # Number of blue cones to identify them later

        # Perform Delaunay triangulation on all cones
        delaunay = Delaunay(all_cones)
        triangles = delaunay.simplices

        midpoints = []
        
        # Process each triangle to find edges between blue and yellow cones
        for triangle in triangles:
            # Display current state of cones and midpoints for this triangle
            # plt.figure(figsize=(8, 6))
            # plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue', label='Blue Cones')
            # plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='yellow', label='Yellow Cones')
            
            for i in range(3):
                # Get the indices of the two points that form an edge
                idx1, idx2 = triangle[i], triangle[(i + 1) % 3]
                
                # Check if idx1 and idx2 belong to different cone groups (blue vs yellow)
                if (idx1 < num_blue and idx2 >= num_blue) or (idx1 >= num_blue and idx2 < num_blue):
                    # Calculate midpoint between the blue and yellow cone
                    point1, point2 = all_cones[idx1], all_cones[idx2]
                    midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
                    midpoints.append(midpoint)

        return midpoints

def filterColors(colorFrame, depthFrame, nedre, ovre):
    # Konverterer framen til hsv farveskalaen så vi bedre kan arbejde med den
    hsvFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2HSV)

    # colormap til depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.1), cv2.COLORMAP_JET)
        
    # Finder farverne i billedet
    altFarve = cv2.inRange(hsvFrame, nedre, ovre)

    # median blur
    altFarve = cv2.medianBlur(altFarve, 5)

    # Filtrerer små hvide steder fra
    altFarve = cv2.morphologyEx(altFarve, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    altFarve = cv2.morphologyEx(altFarve, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))

    # Lægger masken over dybde billedet så vi kun ser på dybden hvor der er gult
    kegleDepth = cv2.bitwise_and(depthFrame, depthFrame, mask = altFarve)

    # creating region of interest with bounding box around cones with information from the color image
    # contours, hierarchy = cv2.findContours(altFarve, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # removing small contours  
    # contours = [contour for contour in contours if cv2.contourArea(contour) >= 1500]

    # boundingBoxes = []
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     boundingBoxes.append((x, y, w, h))
    #     # drawing bounding boxes
    #     if len(boundingBoxes) > 0:
    #         cv2.rectangle(kegleDepth, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # print("new")
    # print(boundingBoxes)
        
    # Dybde segmentering
    depthRanges = [(1, 300), (301, 600), (601, 900), (901, 1200), (1201, 1500), (1501, 1800), (1801, 2100), (2101, 2400), (2401, 2700), (2701, 3000)]
    segmentedImages = []
    bottomPoints = []
    for minD, maxD in depthRanges:
        # Create a binary mask for the current depth range
        depthMask = cv2.inRange(kegleDepth, minD, maxD)
        # Apply the binary mask to the color image
        maskedImage = cv2.bitwise_and(colorFrame, colorFrame, mask=depthMask)
        #finding the coordinates of the cone
        contours, hierarchy = cv2.findContours(depthMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filteredContours = [contour for contour in contours if cv2.contourArea(contour) >= 250]
        combinedContours = combineContours(filteredContours)

        for contour in combinedContours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(maskedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the bottom point of the bounding box
            bottom_point = (x + w // 2, y + h, w)
            bottomPoints.append(bottom_point)
            cv2.circle(maskedImage, (bottom_point[0], bottom_point[1]), 5, (0, 0, 255), -1)
            cv2.putText(maskedImage, "Bottom", (bottom_point[0], bottom_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Calculate the centroid of the bounding box
            cX = x + w // 2
            cY = y + h // 2
            #cv2.putText(maskedImage, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Writes yellow in the middle of the hull
            cv2.putText(maskedImage, f'{minD}; {maxD}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Append the masked image to the list of segmented images
        segmentedImages.append(maskedImage)

        # cv2.imshow(f'Segmented Image {min_d}; {max_d}', masked_image)
        # cv2.waitKey(0)

    # Combine all segmented images into a single image
    combinedImage = np.zeros_like(colorFrame)
    for image in segmentedImages:
        combinedImage = cv2.bitwise_or(combinedImage, image)

    return combinedImage, bottomPoints

def combineContours(contours):
    # Combine two specific contours based on their relative positions
    # Sort contours by their x-coordinate (top to bottom)
    sortedContours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    combinedContours = []

    for contour in range(0, len(sortedContours) -1, 2):
        # Select the top two contours
        contour1 = sortedContours[contour]
        contour2 = sortedContours[contour + 1]

        # Combine the two contours
        combinedContour = np.vstack((contour1, contour2))
        combinedContours.append(combinedContour)
        
    return combinedContours

def chooseSide(side,frame):
    height = frame.shape[0]
    width = frame.shape[1]
    
    if side == "l":
        return frame[0:height, 0:width//2]
    elif side == "r":
        return frame[0:height, width//2:width]
    else:
        return frame

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    alignTo = rs.stream.color
    align = rs.align(alignTo)

    #ready for plotting
    plt.ion()
    plt.figure(figsize=(8, 6))

    try:
        while True:
            if time.time() - startTime > 2.5:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                alignedFrames = align.process(frames)
                colorFrame = alignedFrames.get_color_frame()
                depthFrame = alignedFrames.get_depth_frame()
                if not colorFrame or not depthFrame:
                    continue

                # Convert images to numpy arrays
                colorImage = np.asanyarray(colorFrame.get_data())
                depthImage = np.asanyarray(depthFrame.get_data())

                # Color map af depth image
                depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.1), cv2.COLORMAP_JET)

                gult, bottomPointsG = filterColors(colorImage, depthImage, nedreGul, ovreGul)
                blaa, bottomPointsB = filterColors(colorImage, depthImage, nedreBlaa, ovreBlaa)
                
                cartisianCoordinatesG = []
                for point in bottomPointsG:
                    cartisianCoordinatesG.append(get_cartesian_coordinates(point[0], point[1], point[2], depthImage, gult, focalLength, halfImageWidth))

                cartisianCoordinatesB = []
                for point in bottomPointsB:
                    cartisianCoordinatesB.append(get_cartesian_coordinates(point[0], point[1], point[2], depthImage, blaa, focalLength, halfImageWidth))

                # Calculate midpoints between blue and yellow cones
                midpoints = calculate_midpoints(cartisianCoordinatesB, cartisianCoordinatesG)

                # Display the midpoints
                plt.scatter([point[0] for point in cartisianCoordinatesB], [point[1] for point in cartisianCoordinatesB], color='blue', label='Blue Cones')
                plt.scatter([point[0] for point in cartisianCoordinatesG], [point[1] for point in cartisianCoordinatesG], color='yellow', label='Yellow Cones')
                plt.scatter([point[0] for point in midpoints], [point[1] for point in midpoints], color='red', label='Midpoints')
                plt.legend()
                plt.show()
                plt.pause(0.1)
                plt.clf()

                # Combine the two images
                combinedImage = cv2.bitwise_or(gult, blaa)

                #print("Gul: ", bottomPointsG)
                #print("Blaa: ", bottomPointsB)

                # Show the images
                cv2.imshow('RealSense Depth', depthColormap)
                # cv2.imshow('RealSense Color', color_image)
                cv2.imshow('gult', combinedImage)
                #cv2.imshow('Binary Mask', binary_mask)
                #cv2.imshow('Masked Image', masked_image)

                # cv2.waitKey(0)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()