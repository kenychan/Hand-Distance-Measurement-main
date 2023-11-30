import cv2 
import math
import mediapipe as mp
import numpy as np 
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from direction_speed import MovingDirectionDetector
# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

#initialize direction detector 
DirectionDetector=MovingDirectionDetector()


def map_to_range(value, input_min, input_max, output_min, output_max):
    # Ensure value is within the input range
    value = max(min(value, input_max), input_min)

    # Map the value from the input range to the output range
    input_range = input_max - input_min
    output_range = output_max - output_min

    scaled_value = (value - input_min) / input_range
    mapped_value = output_min + (scaled_value * output_range)

    # Return the mapped value as an integer
    return int(mapped_value)

# Function to calculate distance between two points in 2D space
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


prev_density = None

def calculate_density(width, height, distance, threshold=80):
    global prev_density

    # Normalize the frame size, independent from distance and offset
    norm_width = (width * distance) / (width + distance)
    norm_height = (height * distance) / (height + distance)

    mean_density = norm_width * norm_height

    # Check if there's a drastic change compared to the previous density value
    if prev_density is None or abs(mean_density - prev_density) >= threshold:
        print("Mean density:", mean_density)
        prev_density = mean_density


# Data points for a polynomial regression
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# OpenCV code to capture video from the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 720)  # Set the width of the frame
cap.set(4, 480)  # Set the height of the frame
mylmList = []
img_counter=0
# Main loop to process video frames
while True:
    isopen, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB for Mediapipe processing
    results = hands.process(img)  # Process the frame with Mediapipe Hands
    allHands = []
    h, w, c = frame.shape  # Get the height, width, and number of channels of the frame
    
    # Process each detected hand in the frame
    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            mylmList = []
            xList = []
            yList = []
            
            # Extract landmark points and store them in lists
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([id, px, py])
                xList.append(px)
                yList.append(py)
            

            #paired_list = np.column_stack((xList, yList))#2d lists of hand features
            

            # Calculate bounding box around the hand landmarks
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            
            # Store hand information in a dictionary
            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)
            myHand["type"] = handType.classification[0].label
            #if you dont flip the image
            ''' if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
            else:
                        myHand["type"] = "Right"'''
            allHands.append(myHand)
            # Draw landmarks and bounding box on the frame
            mpdraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
            cv2.putText(frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            # Calculate and display the distance between two specific landmarks of the hand
            if mylmList != 0:
                try:
                    x, y = mylmList[5][1], mylmList[5][2]
                    x2, y2 = mylmList[17][1], mylmList[17][2]
                    dis = calculate_distance(x, y, x2, y2)
         
                    A, B, C = coff
                    distanceCM = A * dis**2 + B * dis + C #distance from camera to hand
                    #print(distanceCM)
                    cv2.rectangle(frame, (xmax - 80, ymin - 80), (xmax + 20, ymin - 20), (255, 0, 255), cv2.FILLED)
                    cv2.putText(frame, f"{int(distanceCM)}cm", (xmax - 80, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                    calculate_density(boxW, boxH,distanceCM)
                    DirectionDetector.moving_direction(cx,cy,130)
                except:
                    pass
    
    # Display the frame with annotations
    cv2.imshow('Hand Distance Measurement', frame)
    
    # Exit the loop if 'q' key is pressed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break






