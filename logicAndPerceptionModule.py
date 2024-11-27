import math
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import serial

startTime = time.time()

display_plot = False

###PERCEPTION MODULE###
#grænseværdier for gul farve i HSV
nedreGul = (18,90,115)
ovreGul = (32,255,255)

#grænseværdier for blå farve i HSV
nedreBlaa = (100,230,70)
ovreBlaa = (115,255,255)

#grænseværdier for orange farve i HSV
nedreOrange = (0,150,170)
ovreOrange = (17,255,255)

#Orange cones count
orangeFrameCount = 0
orangeNotSeenCount = 0
orangeSeen = False
######################

###CONTROL MODULE###
speed = 100
maxTurnAngle = 30 #Max turn angle(degrees) from middle to left/middle to right
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1) #Arduino
####################

###PID VALUES####
pControlValue = 0.8
iControlValue = 0
dControlValue = 0

integral_error = 0.0
previous_error = 0.0
#################



def close_plot(event):
                    if event.key == 'q': 
                        plt.close()

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
    d = (np.nanmean(depth_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]) + 50.0)
    if d > 430:
        distance=math.sqrt(d**2 - 430**2)
    else:
        distance=0

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

    if len(all_cones) == 0:
        midpoints = [(10,0)]
        return midpoints

    
    if len(all_cones)>0 and (len(blue_cones)==0 or len(yellow_cones)==0):  
        #IF no midpoints exist, but it sees a cone, it will avoid it
        midpoints = []
        if len(blue_cones)>0:
            midpoints.append(((blue_cones[0][0]),(blue_cones[0][1]+450))) 
        else:
            midpoints.append(((yellow_cones[0][0]),(yellow_cones[0][1]-450)))
        return midpoints
    
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
        
        # midpoints.insert(0, (0, 0))
        return midpoints

def listOfCartisianCoords(bottomPoints, depthImage, kegleFrame):
    
    FOV = 87.0
    width = 640
    halfImageWidth = width / 2
    focalLength = width / (2 * math.tan(math.radians(FOV / 2)))
    
    cartisianCoordinates = []
    for point in bottomPoints:
        cartisianCoordinates.append(get_cartesian_coordinates(point[0], point[1], point[2], depthImage, kegleFrame, focalLength, halfImageWidth))
    return cartisianCoordinates

def filterColors(colorFrame, nedre, ovre, blursize):
    # Konverterer framen til hsv farveskalaen så vi bedre kan arbejde med den
    hsvFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2HSV)
        
    # Finder farverne i billedet
    mask = cv2.inRange(hsvFrame, nedre, ovre)

    # median blur
    mask = cv2.medianBlur(mask, blursize)

    # Filtrerer små hvide steder fra
    # altFarve = cv2.morphologyEx(altFarve, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))

    return mask

def depthSegmentation(binaryImage, depthFrame, colorFrame, combineTheContours, text):
    kegleDepth = cv2.bitwise_and(depthFrame, depthFrame, mask = binaryImage)

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
    
        
    # Dybde segmentering
    depthRangesList = [(1, 300), (301, 600), (601, 900), (901, 1200), (1201, 1500), (1501, 1800), (1801, 2100), (2101, 2400), (2401, 2700), (2701, 3000)]
    depthRanges = np.array(depthRangesList)
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
        if combineTheContours:
            combinedContours = combineContours(filteredContours)
        else:
            combinedContours = filteredContours

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
            cv2.putText(maskedImage, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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

def calculate_curvature(spline, x_val):
    # Calculate the curvature of the spline
    dx = spline.derivative(nu=1)
    ddx = spline.derivative(nu=2)
    curvature = np.abs(ddx(x_val)) / (1 + dx(x_val)**2)**1.5
    return curvature

def predict_curvature(coords):
    if len(coords) < 2:
        print("Insufficient points to calculate curvature.")
        return 0.0, None  # Return 0.0 if there are too few points

    # Ensure coords is a list of tuples with two elements each
    coords = [(x, y) for x, y in coords if len((x, y)) == 2]

    # Remove duplicate x values while preserving order
    coords_dict = {x: y for x, y in coords}
    unique_coords = list(coords_dict.items())

    # Sort the coordinates by x values
    sorted_coords = sorted(unique_coords, key=lambda coord: coord[0])
    x_coords, y_coords = zip(*sorted_coords)

    spline = CubicSpline(x_coords, y_coords)

    x_smooth = np.linspace(min(x_coords), max(x_coords), 10)

    # Calculates the curvature of the spline at each point
    avg_curvature = np.average([calculate_curvature(spline, x) for x in x_smooth])
    return avg_curvature, spline

def calculate_speed(curvature):
    # Linear interpolation from v_min at max_curvature to v_max at curvature = 0
    global speed
    v_min = 100
    v_max = 140
    max_curvature = 0.0025
    speed = v_min + (v_max - v_min) * (1- (curvature/max_curvature))

def plotPointsOgMidpoints(blaaCartisianCoordinates, guleCartisianCoordinates, midpoints, spline):
    plt.scatter([point[0] for point in blaaCartisianCoordinates], [point[1] for point in blaaCartisianCoordinates], color='blue', label='Blue Cones')
    plt.scatter([point[0] for point in guleCartisianCoordinates], [point[1] for point in guleCartisianCoordinates], color='yellow', label='Yellow Cones')
    plt.scatter([point[0] for point in midpoints], [point[1] for point in midpoints], color='red', label='Midpoints')

    # Generate points on the spline
    x_coords = [point[0] for point in midpoints]
    # x_coords = np.insert(x_coords, 0,[0.0,0.0]) # Insert cars position as the first point
    x_smooth = np.linspace(min(x_coords), max(x_coords), 10)
    
    y_smooth = spline(x_smooth)
    #y_smooth[0] = 0.0
    # y_smooth = np.insert(y_smooth, 0,0.0) # Insert cars position as the first point
    
    # Plot the spline
    plt.plot(x_smooth, y_smooth, color='green', label='Spline')

    plt.xlim(0, 3000)
    plt.ylim(-1500, 1500)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    plt.pause(0.1)
    plt.clf()


###CONTROL MODULE####
def speedAngleArduino(localspeed,angle):
    #localspeed is the speed set in this function, and not the global
    localspeed = str(localspeed).zfill(3)
    angle = str(angle).zfill(3)
    val = f"{localspeed}{angle}\n"
    # print("Value send to arduino",val)
    arduino.write(bytes(val, 'utf-8'))
    # time.sleep(0.05)

def deg2turnvalue(deg):
    return 90 + deg*90/maxTurnAngle # 37 is the max turn angle in degrees      -      90 is the middle for the servo(0-180)

def steerToAngleOfCoordinate(currentxy, targetxy):
    global integral_error, previous_error

    #Deviation in x and y
    dx = targetxy[0] - currentxy[0]
    dy = targetxy[1] - currentxy[1]
    
    #Distance from target
    # distance = (dx**2 + dy**2)**0.5
    # print("Distance from target x,y:",distance,"mm")

    #Calculate the angle error
    angleError = math.atan2(dx,dy) # Angle in radians
    angleError = 90-math.degrees(angleError) # Convert to degrees and convert by 90 deg to get correct angle

    #PID
    proportionalValue = pControlValue * angleError #P
    integral_error += angleError #I error
    integralValue = iControlValue * integral_error #I
    derivativeValue = angleError - previous_error #D
    # print("PID CONTROLLER: Proportional Value:",proportionalValue,"Integral Value:",integralValue,"Derivative Value:",derivativeValue)
    
    #Avoid integral_error overflow
    if integral_error > 300:
        integral_error = 300

    #Remember the current error for next iteration
    previous_error = angleError

    #sums P, I and D
    steeringAngle = (proportionalValue + integralValue + derivativeValue)*-1 #Negative to get correct direction
    
    if steeringAngle > maxTurnAngle:
        steeringAngle = maxTurnAngle
        # print("max turn angle achieved.")
    elif steeringAngle < -maxTurnAngle:
        steeringAngle = -maxTurnAngle
        # print("min turn angle achieved.")

    
    # print("steeringAngle:",steeringAngle)
    # print("ServoValue:",int(deg2turnvalue(steeringAngle))) #SKAL MÅSKE KONVERTERES TIL INT
    
    speedAngleArduino(speed,int(deg2turnvalue(steeringAngle)))
    #print("AngleError:",angleError,"Angle send to arduino:",int(deg2turnvalue(steeringAngle)))

def orangeDetection(orangeCones):
    global orangeFrameCount, orangeSeen, orangeNotSeenCount, speed
    if (len(orangeCones) > 0) and not orangeSeen:
        orangeFrameCount += 1
        
        if orangeFrameCount == 10: #If orange cone is seen for 10 frames
            print("ORANGE CONE DETECTED - INITIATING STOP")
            orangeSeen = True
    if orangeSeen:
        if len(orangeCones)==0:
            orangeNotSeenCount += 1
            
            if orangeNotSeenCount > 5: #If orange cone is not seen for 100 frames
                speedAngleArduino(90,90) #Stops car
                print("CAR STOPPED - STOP LINE REACHED")
                return True
    return False 
timeNow = 0
timeNowTotal = 0
def main():
    global timeNow, timeNowTotal
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    alignTo = rs.stream.color
    align = rs.align(alignTo)

    # Ready for plotting
    if display_plot:
        plt.ion()
        plt.figure(figsize=(8, 6))

    try:
        while True:
            try:
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
                    print("timestamp 1:", timeNow - time.time())
                    timeNow = time.time()

                    # Color map af depth image
                    #depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.1), cv2.COLORMAP_JET)

                    # Filter colors to get masks for yellow and blue cones
                    print("timestamp 1:", timeNow - time.time())
                    timeNow = time.time()
                    gulMask = filterColors(colorImage, nedreGul, ovreGul, 5)
                    print("timestamp 2:", timeNow - time.time())
                    timeNow = time.time()
                    blaaMask = filterColors(colorImage, nedreBlaa, ovreBlaa, 11)
                    print("timestamp 3:", timeNow - time.time())
                    timeNow = time.time()
                    orangeMask = filterColors(colorImage, nedreOrange, ovreOrange, 11)
                    
                    print("timestamp 4:", timeNow - time.time())
                    timeNow = time.time()
                    # Depth segmentation to differentiate overlapping cones and get the bottom points of the cones
                    guleKegler, guleBottomPoints = depthSegmentation(gulMask, depthImage, colorImage, True, "Gul")
                    
                    print("timestamp 5:", timeNow - time.time())
                    timeNow = time.time()
                    blaaKegler, blaaBottomPoints = depthSegmentation(blaaMask, depthImage, colorImage, True, "Blaa")
                    
                    print("timestamp 6:", timeNow - time.time())
                    timeNow = time.time()
                    orangeKegler, orangeBottomPoints = depthSegmentation(orangeMask, depthImage, colorImage, False, "Orange")

                    print("timestamp 7:", timeNow - time.time())
                    timeNow = time.time()

                    # Convert the pixel coordinates of the bottom points to Cartesian coordinates
                    guleCartisianCoordinates = listOfCartisianCoords(guleBottomPoints, depthImage, guleKegler)
                    blaaCartisianCoordinates = listOfCartisianCoords(blaaBottomPoints, depthImage, blaaKegler)
                    orangeCartisianCoordinates = listOfCartisianCoords(orangeBottomPoints, depthImage, orangeKegler)

                    # Calculate midpoints between blue and yellow cones
                    midpoints = calculate_midpoints(blaaCartisianCoordinates, guleCartisianCoordinates)
                    

                    # Calculate the curvature of the spline
                    max_curvature, spline = predict_curvature(midpoints)
                    # print(f'Maximum curvature: {max_curvature}')

                    # Check if the spline is not None before using itd
                    if display_plot and spline is not None:
                        plotPointsOgMidpoints(blaaCartisianCoordinates, guleCartisianCoordinates, midpoints, spline)
                    

                    # Combine the two images
                    # combinedImage1 = cv2.bitwise_or(guleKegler, blaaKegler)
                    # combinedImage = cv2.bitwise_or(combinedImage1, orangeKegler)

                    # # Show the images
                    # cv2.imshow('RealSense Depth', depthColormap)
                    # cv2.imshow('Combined blue and yellow', combinedImage)
                    # cv2.imshow('Orange', orangeMask)
                    
                    # fig=plt.gcf()
                    # fig.canvas.mpl_connect('key_press_event', close_plot)
                    
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
                    #CONTROL MODULE
                    if len(midpoints) > 0:
                        steerToAngleOfCoordinate([0,0],midpoints[0]) #Car position and target position
                    
                    if orangeDetection(orangeCartisianCoordinates):
                        break
                    
                    #Printing hz of python code
                    print("FPS: ", 1.0 / (time.time() - timeNowTotal))
                    timeNowTotal = time.time()
            except:
                print("Error in main loop")
                    
                    
    finally:
        # Stop streaming
        # pipeline.stop()
        # cv2.destroyAllWindows()
        print("Program ended")

main()
