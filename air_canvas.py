import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math

class AirCanvas:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Drawing parameters
        self.drawing = False
        self.brush_thickness = 10
        self.eraser_thickness = 50
        
        # Colors
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255) ]
        self.color_names = ['BLUE', 'GREEN', 'RED', 'YELLOW']
        self.current_color = (255, 0, 0)  # Default blue
        
        # Canvas
        self.canvas = None
        self.x_prev, self.y_prev = 0, 0
        
        # Points for drawing
        self.blue_points = [deque(maxlen=1024)]
        self.green_points = [deque(maxlen=1024)]
        self.red_points = [deque(maxlen=1024)]
        self.yellow_points = [deque(maxlen=1024)]
        self.magenta_points = [deque(maxlen=1024)]
        
        # Color point arrays
        self.color_points = [self.blue_points, self.green_points, self.red_points, 
                           self.yellow_points]
        self.color_index = 0
        
    def setup_ui(self, frame):
        """Setup the user interface with color palette and controls"""
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), self.colors[0], -1)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), self.colors[1], -1)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), self.colors[2], -1)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), self.colors[3], -1)
        
        
        # Labels
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        return frame
    
    def get_finger_positions(self, landmarks):
        """Extract finger tip positions from hand landmarks"""
        # Index finger tip (landmark 8)
        index_finger_tip = landmarks.landmark[8]
        # Middle finger tip (landmark 12)
        middle_finger_tip = landmarks.landmark[12]
        # Ring finger tip (landmark 16)
        ring_finger_tip = landmarks.landmark[16]
        # Pinky tip (landmark 20)
        pinky_tip = landmarks.landmark[20]
        # Thumb tip (landmark 4)
        thumb_tip = landmarks.landmark[4]
        
        return {
            'index': (int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)),
            'middle': (int(middle_finger_tip.x * 640), int(middle_finger_tip.y * 480)),
            'ring': (int(ring_finger_tip.x * 640), int(ring_finger_tip.y * 480)),
            'pinky': (int(pinky_tip.x * 640), int(pinky_tip.y * 480)),
            'thumb': (int(thumb_tip.x * 640), int(thumb_tip.y * 480))
        }
    
    def is_finger_up(self, landmarks, finger_tip_id, finger_pip_id):
        """Check if a finger is up by comparing tip and PIP joint positions"""
        return landmarks.landmark[finger_tip_id].y < landmarks.landmark[finger_pip_id].y
    
    def detect_gesture(self, landmarks):
        """Detect hand gestures for drawing and control"""
        # Get finger positions
        fingers = self.get_finger_positions(landmarks)
        
        # Check which fingers are up
        thumb_up = self.is_finger_up(landmarks, 4, 3)
        index_up = self.is_finger_up(landmarks, 8, 6)
        middle_up = self.is_finger_up(landmarks, 12, 10)
        ring_up = self.is_finger_up(landmarks, 16, 14)
        pinky_up = self.is_finger_up(landmarks, 20, 18)
        
        # Drawing gesture: Only index finger up
        if index_up and not middle_up and not ring_up and not pinky_up:
            return 'draw', fingers['index']
        
        # Selection gesture: Index and middle finger up
        elif index_up and middle_up and not ring_up and not pinky_up:
            return 'select', fingers['index']
        
        # No gesture
        else:
            return 'none', None
    
    def handle_color_selection(self, x, y):
        """Handle color selection from the palette"""
        if y <= 65:
            if 40 <= x <= 140:  # Clear button
                self.clear_canvas()
            elif 160 <= x <= 255:  # Blue
                self.color_index = 0
                self.current_color = self.colors[0]
            elif 275 <= x <= 370:  # Green
                self.color_index = 1
                self.current_color = self.colors[1]
            elif 390 <= x <= 485:  # Red
                self.color_index = 2
                self.current_color = self.colors[2]
            elif 505 <= x <= 600:  # Yellow
                self.color_index = 3
                self.current_color = self.colors[3]
            
    
    def clear_canvas(self):
        """Clear the canvas and all drawing points"""
        for color_point in self.color_points:
            color_point.clear()
        # Re-initialize with empty deques
        self.blue_points = [deque(maxlen=1024)]
        self.green_points = [deque(maxlen=1024)]
        self.red_points = [deque(maxlen=1024)]
        self.yellow_points = [deque(maxlen=1024)]
        
        self.color_points = [self.blue_points, self.green_points, self.red_points, 
                           self.yellow_points]
    
    def draw_on_canvas(self, frame):
        """Draw all the stored points on the frame"""
        # Draw all color points
        for i, color_point in enumerate(self.color_points):
            for points in color_point:
                # Convert deque to list and ensure we have valid points
                point_list = list(points)
                for j in range(1, len(point_list)):
                    if point_list[j-1] is None or point_list[j] is None:
                        continue
                    # Ensure points are tuples of integers
                    try:
                        pt1 = tuple(map(int, point_list[j-1]))
                        pt2 = tuple(map(int, point_list[j]))
                        cv2.line(frame, pt1, pt2, self.colors[i], self.brush_thickness)
                    except (TypeError, ValueError):
                        continue
        return frame
    
    def run(self):
        """Main function to run the Air Canvas application"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        
        print("Air Canvas Started!")
        print("Instructions:")
        print("- Point with index finger to draw")
        print("- Use index + middle finger to select colors")
        print("- Click on color buttons to change colors")
        print("- Click CLEAR to clear canvas")
        print("- Press 'q' to quit")
        
        prev_gesture = 'none'
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Setup UI
            frame = self.setup_ui(frame)
            
            # Process hand detection
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture
                    gesture, position = self.detect_gesture(hand_landmarks)
                    
                    if gesture == 'draw' and position:
                        x, y = position
                        
                        # Ensure we have a valid position tuple
                        if isinstance(position, tuple) and len(position) == 2:
                            # Add point to current color's point list
                            if len(self.color_points[self.color_index]) == 0:
                                self.color_points[self.color_index].append(deque(maxlen=512))
                            
                            self.color_points[self.color_index][-1].append(position)
                            
                            # Draw circle at current position
                            cv2.circle(frame, position, self.brush_thickness, self.current_color, -1)
                    
                    elif gesture == 'select' and position:
                        x, y = position
                        self.handle_color_selection(x, y)
                        
                        # Show selection circle
                        cv2.circle(frame, position, 15, (0, 255, 0), 2)
                    
                    # If gesture changed from draw to something else, start new stroke
                    if prev_gesture == 'draw' and gesture != 'draw':
                        self.color_points[self.color_index].append(deque(maxlen=512))
                    
                    prev_gesture = gesture
            
            # Draw all the stored points
            frame = self.draw_on_canvas(frame)
            
            # Show current color indicator
            cv2.putText(frame, f"Current: {self.color_names[self.color_index]}", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.current_color, 2)
            
            # Display the frame
            cv2.imshow('Air Canvas', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to start the Air Canvas application"""
    try:
        canvas = AirCanvas()
        canvas.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required libraries installed.")
        print("Required libraries: opencv-python, mediapipe, numpy")

if __name__ == "__main__":
    main()
