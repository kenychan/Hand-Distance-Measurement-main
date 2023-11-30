import math

class MovingDirectionDetector:
    def __init__(self):
        self.prev_x = None
        self.prev_y = None
        self.direct_arr=[]
    def moving_direction(self, cx, cy, speed_threshold):
        # If it's the first time or if prev_x, prev_y are not initialized
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = cx
            self.prev_y = cy
            print("Initializing previous coordinates.")
            return
        
        
        # Calculate the change in x and y
        dx = cx - self.prev_x
        dy = cy - self.prev_y

        # Calculate the angle in radians between the old and new points
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360  # Convert negative angles to positive
        

        # Calculate the distance between previous and current positions
        distance = math.sqrt((cx - self.prev_x) ** 2 + (cy - self.prev_y) ** 2)
        
        # Calculate speed (distance per unit time, for example, per frame)
        speed = distance  # Assuming 1 unit of time for simplicity
        
        # Check if the speed exceeds the threshold for generating directions
        if speed >= speed_threshold:
            # Determine direction based on change in x and y coordinates
            direction = ""
            if 45 < angle_deg <= 135: #for some reason, probably cuz the coordinates are inverted, up and down are inverted
                direction = "down"
            elif 315 < angle_deg <= 360 or 0 <= angle_deg <= 45:
                direction = "right"
            elif 225 < angle_deg <= 315:
                direction = "up"
            elif 135 < angle_deg <= 225:
                direction = "left"

            self.direct_arr.append(direction)
            # Print direction and speed
            if direction and len(self.direct_arr)==1:
                print(f"Moving {direction} at a speed of {speed}")
                self.direct_arr=[]
            else:
                self.direct_arr=[]
                print("Stationary")
        
        # Update previous coordinates
        self.prev_x = cx
        self.prev_y = cy