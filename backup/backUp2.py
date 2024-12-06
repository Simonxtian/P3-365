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
display_images = False
laps = 10

###PERCEPTION MODULE###
#gr�nsev�rdier for gul farve i HSV
nedreGul = (18,90,115)
ovreGul = (32,255,255)

#gr�nsev�rdier for bl� farve i HSV
nedreBlaa = (100,190,60)
ovreBlaa = (115,255,255)

#gr�nsev�rdier for orange farve i HSV
nedreOrange = (2,200,170)
ovreOrange = (17,255,255)


#Orange cones count
orangeFrameCount = 0
orangeNotSeenCount = 0
orangeSeen = False
allLapsCompleted = False
tenSteps = 0
lapCount = 0
######################

###CONTROL MODULE###
speed = 90
maxTurnAngle = 30 #Max turn angle(degrees) from middle to left/middle to right
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1) #Arduino
####################

###PID GAIN VALUES####
kP = 1.3
kI = 0.0003
kD = 0.7

integralError = 0.0
previousError = 0.0
#################

###FPS FIX###
timeNow = 0
timeNowTotal = 0
stopTime = 0
#############

def close_plot(event):
                    if event.key == 'q': 
                        plt.close()

def get_cartesian_coordinates(x, y, w, depth_image, img):
    # Calculate horizontal angle of the object from the center of the image
    cx = 210.0619354248047 # Principal point x-coordinate
    cy = 120.53054809570312 # Principal point y-coordinate
    focalLength = 210.0619354248047 # Focal length, in pixels
        
    # Camera tilt in degrees
    camera_tilt = np.radians(15)

    # Camera intrinsics
    K = np.array([[focalLength, 0, cx], [0, focalLength, cy], [0, 0, 1]])
    # Rotation matrix
    R = np.array([  [1, 0, 0,], 
                    [0, np.cos(camera_tilt), -np.sin(camera_tilt)], 
                    [0, np.sin(camera_tilt), np.cos(camera_tilt)]   ])
    # Projection matrix
    P = np.dot(K, R)
    P_inv = np.linalg.inv(P)
    # NTS: Include distance calculation

    # Get distance from the depth frame
    bbox = (x-w//5, y-w*2//5, w*2//5, w*2//5)
    # Draw bbox
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
    # average distance in the bounding box in mm
    d = (np.nanmean(depth_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]) + 50.0)

    world_coords = (np.dot(P_inv, np.array([x, y, 1])))
    norm = np.linalg.norm(world_coords)
    world_coords = world_coords/norm
    world_coords = world_coords*d

    y_cart = world_coords[0]
    x_cart = world_coords[2]

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
        #If no cones are seen, a midpoint is created straight in front of the car
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
    cartisianCoordinates = []
    for point in bottomPoints:
        cartisianCoordinates.append(get_cartesian_coordinates(point[0], point[1], point[2], depthImage, kegleFrame))
    return cartisianCoordinates

def filterColors(colorFrame, nedre, ovre):
    # Konverterer framen til hsv farveskalaen s� vi bedre kan arbejde med den
    hsvFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2HSV)
        
    # Finder farverne i billedet
    mask = cv2.inRange(hsvFrame, nedre, ovre)

    # median blur
    mask = cv2.medianBlur(mask, 5)

    # Filtrerer sm� hvide steder fra
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask

def depthSegmentation(binaryImage, depthFrame, colorFrame, combineTheContours, text):
    global timeNow
    # Apply the binary mask to the depth frame
    kegleDepth = cv2.bitwise_and(depthFrame, depthFrame, mask=binaryImage)
    
    depthRanges = [(i, i + 300 - 1) for i in range(1, 3001, 300)]
    # Initialize variables
    bottomPoints = []
    
    # Pre-create an output image
    combinedImage = np.zeros_like(colorFrame)
    
    # Loop through ranges, this part can still use for-loop due to sequential bounding
    for minD, maxD in depthRanges:
        depthMask = (kegleDepth >= minD) & (kegleDepth <= maxD)
        depthMask = depthMask.astype(np.uint8) * 255  # Convert to binary mask
        maskedImage = cv2.bitwise_and(colorFrame, colorFrame, mask=depthMask)
    
        contours, _ = cv2.findContours(depthMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filteredContours = [contour for contour in contours if cv2.contourArea(contour) >= 50]
        
        # Combine contours if required
        if combineTheContours:
            filteredContours = combineContours(filteredContours)
           
        #Sort out the contours that are too small
        filteredContours= [contour for contour in filteredContours if cv2.contourArea(contour) >= 100]
        
        for contour in filteredContours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(maskedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

            bottom_point = (x + w // 2, y + h, w)
            bottomPoints.append(bottom_point)
            cv2.circle(maskedImage, (bottom_point[0], bottom_point[1]), 5, (0, 0, 255), -1)
            cv2.putText(maskedImage, "Bottom", (bottom_point[0], bottom_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cX = x + w // 2
            cY = y + h // 2
            cv2.putText(maskedImage, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        combinedImage = cv2.bitwise_or(combinedImage, maskedImage) #TAR LANG TID
        
    
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

def calculate_speed(curvature):
    global speed
    # Linear interpolation from v_min at max_curvature to v_max at curvature = 0
    v_min=122 #115
    v_max=122 #120
    max_curvature=0.002
    #0.002
    try:
        speed = int(v_min + (v_max - v_min) * (1- (curvature/max_curvature)))
        
        if speed > v_max:
            speed = v_max
        elif speed < v_min:
            speed = v_min

        if curvature == 0:
            #When no cones are detected, the car will drive at a constant slow speed
            speed = v_min
        
    except:
        print("Division by zero in calculate_speed")

def predict_curvature(coords):
    coords.append((0.0,0.0))  # Add a point at the origin of the car
    if len(coords) < 2:
        return 0.0, None  #Return 0.0 if there are too few points
    
    
    # Ensure coords is a list of tuples with two elements each
    coords = [(x, y) for x, y in coords if len((x, y)) == 2]
    
    

    # Remove duplicate x values while preserving order
    coords_dict = {x: y for x, y in coords}
    unique_coords = list(coords_dict.items())

    # Sort the coordinates by x values
    sorted_coords = sorted(unique_coords, key=lambda coord: coord[0])
    
    x_coords, y_coords = zip(*sorted_coords)
    

    spline = CubicSpline(x_coords, y_coords, bc_type=((1,0),'natural'))
    x_smooth = np.linspace(min(x_coords), max(x_coords), 10)
    curvatures = [calculate_curvature(spline, x) for x in x_smooth]
    max_curvature = np.max(curvatures)

    return max_curvature, spline

def plotPointsOgMidpoints(blaaCartisianCoordinates, guleCartisianCoordinates, midpoints, spline):
    plt.scatter([point[0] for point in blaaCartisianCoordinates], [point[1] for point in blaaCartisianCoordinates], color='blue', label='Blue Cones')
    plt.scatter([point[0] for point in guleCartisianCoordinates], [point[1] for point in guleCartisianCoordinates], color='yellow', label='Yellow Cones')
    plt.scatter([point[0] for point in midpoints], [point[1] for point in midpoints], color='red', label='Midpoints')

    # Generate points on the spline
    x_coords = [point[0] for point in midpoints]
    x_smooth = np.linspace(min(x_coords), max(x_coords), 10)
    y_smooth = spline(x_smooth)
    
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
    arduino.write(bytes(val, 'utf-8'))

def deg2turnvalue(deg):
    return 90 + deg*90/maxTurnAngle # 37 is the max turn angle in degrees      -      90 is the middle for the servo(0-180)

def steerToAngleOfCoordinate(currentxy, targetxy):
    global integralError, previousError, speed

    #Deviation in x and y
    dx = targetxy[0] - currentxy[0]
    dy = targetxy[1] - currentxy[1]
    
    #Calculate the angle error
    angleError = math.atan2(dx,dy) # Angle in radians
    angleError = 90-math.degrees(angleError) # Convert to degrees and convert by 90 deg to get correct angle

    #print(angleError)

    #PID
    proportionalValue = kP * angleError #P
    integralError += angleError #I error
    integralValue = kI * integralError #I
    derivativeValue = kD*(angleError - previousError) #D
    
    #Avoid integralError overflow
    if integralError > 500:integralError
        integralError = 500
    elif integralError < -500:
        integralError = -500

    #Remember the current error for next iteration
    previousError = angleError

    #sums P, I and D
    steeringAngle = (proportionalValue + integralValue + derivativeValue)*-1 #Negative to get correct direction
    
    if steeringAngle > maxTurnAngle:
        steeringAngle = maxTurnAngle
       
    elif steeringAngle < -maxTurnAngle:
        steeringAngle = -maxTurnAngle
    
    speedAngleArduino(speed,int(deg2turnvalue(steeringAngle)))

def stopLineDetection(orangeCones):
    global orangeFrameCount, orangeSeen, orangeNotSeenCount, speed, stopTime, allLapsCompleted, tenSteps, lapCount
    if (len(orangeCones) > 0) and not orangeSeen:
        orangeFrameCount += 1
        
        if orangeFrameCount == 5: #If orange cone is seen for 5 frames
            print("ORANGE CONE DETECTED - INITIATING STOP")
            orangeSeen = True

    if orangeSeen:
        if len(orangeCones)==0:
            orangeNotSeenCount += 1
            
            if orangeNotSeenCount > 5: #If orange cone is not seen for 5 frames
                #Descelerate over a period of 2 seconds while running main function
                
                    lapCount += 1
                    

                    if lapCount >= laps:
                        if not allLapsCompleted:
                            allLapsCompleted = True
                            tenSteps = -(speed-90)//10
                            print(lapCount, "LAPS COMPLETED - STOPPING CAR")

                        if time.time() - stopTime > 0.2:
                            stopTime = time.time()
                            speed += tenSteps

                        if speed < 91:
                            speedAngleArduino(90,90)
                            return True
                    else:   
                        orangeSeen = False
                        orangeFrameCount = 0
                        orangeNotSeenCount = 0
            
    return False 

def main():
    global timeNow, timeNowTotal, allLapsCompleted
    #FPS values
    totalFrames = 0
    averageFPS = 0
    lastFPS = 0
    maxFPS = 0
    minFPS = 1000
    cumulativeFPS = 0
    lowFPSCount = 0
    

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)

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
                if time.time() - startTime > 5:
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
                    
                    # Filter colors to get masks for yellow and blue cones
                    
                    gulMask = filterColors(colorImage, nedreGul, ovreGul)
                   
                    blaaMask = filterColors(colorImage, nedreBlaa, ovreBlaa)
                    
                    orangeMask = filterColors(colorImage, nedreOrange, ovreOrange)
                    
                    
                    # Depth segmentation to differentiate overlapping cones and get the bottom points of the cones
                    guleKegler, guleBottomPoints = depthSegmentation(gulMask, depthImage, colorImage, True, "Gul")
                    
                    
                    blaaKegler, blaaBottomPoints = depthSegmentation(blaaMask, depthImage, colorImage, True, "Blaa")
                    
                    
                    orangeKegler, orangeBottomPoints = depthSegmentation(orangeMask, depthImage, colorImage, False, "Orange")


                    # Convert the pixel coordinates of the bottom points to Cartesian coordinates
                    guleCartisianCoordinates = listOfCartisianCoords(guleBottomPoints, depthImage, guleKegler)
                    blaaCartisianCoordinates = listOfCartisianCoords(blaaBottomPoints, depthImage, blaaKegler)
                    orangeCartisianCoordinates = listOfCartisianCoords(orangeBottomPoints, depthImage, orangeKegler)

                    # Calculate midpoints between blue and yellow cones
                    midpoints = calculate_midpoints(blaaCartisianCoordinates, guleCartisianCoordinates)
                    

                    # Calculate the curvature of the spline
                    averageCurvature, spline = predict_curvature(midpoints)
                    

                    # Check if the spline is not None before using itd
                    if display_plot and spline is not None:
                        plotPointsOgMidpoints(blaaCartisianCoordinates, guleCartisianCoordinates, midpoints, spline)
                    
                    if display_images:
                        #DISPLAY NEDENFOR 
                        # Color map af depth image UDKOMMENTER
                        depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.1), cv2.COLORMAP_JET)
                        # Combine the two images
                        combinedImage1 = cv2.bitwise_or(guleKegler, blaaKegler)
                        combinedImage = cv2.bitwise_or(combinedImage1, orangeKegler)
                        # Show the images
                        cv2.imshow('RealSense Depth', depthColormap)
                        cv2.imshow('Combined blue and yellow', combinedImage)
                        cv2.imshow('Orange', orangeMask)
                        fig=plt.gcf()
                        fig.canvas.mpl_connect('key_press_event', close_plot)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if stopLineDetection(orangeCartisianCoordinates):
                        #This has to stop before steerToAngleOfCoordinate is called (otherwise it will not stop, because of serial timeout in arduino)
                        break
                    elif not allLapsCompleted:
                        calculate_speed(averageCurvature)
                
                    #CONTROL MODULE
                    if len(midpoints) > 0:
                        steerToAngleOfCoordinate([0,0],midpoints[0]) #Car position and target position
                    
                    #Printing hz of python code
                    FPS = 1.0 / (time.time() - timeNowTotal)
                    totalFrames += 1
                    cumulativeFPS += FPS
                    averageFPS = cumulativeFPS/totalFrames

                    if lastFPS>maxFPS:
                        maxFPS = lastFPS
                    elif (lastFPS < minFPS or minFPS == 1000) and lastFPS != 0:
                        minFPS = lastFPS

                    if FPS < 30:
                        lowFPSCount += 1
                        
                    lowFPSPercentange = (lowFPSCount/totalFrames)*100

                    #print("minFPS: ", minFPS, "maxFPS: ", maxFPS, "averageFPS: ", averageFPS, "lowFPSPercentange: ", lowFPSPercentange)

                    lastFPS = FPS
                    




                    timeNowTotal = time.time()
            except:
                print("Error in main loop")
                    
    finally:
        # Stop streaming
        speedAngleArduino(90,90)#MAKE SURE THE CAR STOPS
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Program ended")

main()