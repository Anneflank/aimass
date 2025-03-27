import cv2
import numpy as np
import pyautogui
import time
from pynput import mouse, keyboard
import mss
import math
import threading

# --- Configuration ---
# ROI (Region of Interest) settings - Box around the center of the screen
ROI_WIDTH = 400
ROI_HEIGHT = 400

# Target Color Range (HSV) - Adjusted for Red Color
LOWER_COLOR = np.array([0, 120, 70])   # Lower bound for HSV color (Red)
UPPER_COLOR = np.array([10, 255, 255])  # Upper bound for HSV color (Red)

# Aiming settings
ACTIVATION_KEY = keyboard.Key.alt_l # Key to hold down to activate aim assist (Left Alt)
MOVEMENT_SMOOTHING = 0.15  # Fraction of distance to move each step (lower = smoother/slower)
MIN_CONTOUR_AREA = 50      # Ignore contours smaller than this area
AIM_LOOP_DELAY = 0.01      # Delay between aiming adjustments (seconds)
TARGET_SCAN_DELAY = 0.03   # Delay between screen scans when key is held

# --- Globals ---
screen_width, screen_height = pyautogui.size()
roi_left = (screen_width - ROI_WIDTH) // 2
roi_top = (screen_height - ROI_HEIGHT) // 2
roi = {'top': roi_top, 'left': roi_left, 'width': ROI_WIDTH, 'height': ROI_HEIGHT}

aim_assist_active = False
target_lock = None # Store the currently locked target position

# --- Functions ---

def get_target_position_color(sct):
    """Captures ROI, finds the largest contour within the color range."""
    try:
        # Capture the ROI
        img = np.array(sct.grab(roi))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Convert to BGR

        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply GaussianBlur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Create a mask for the target color
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter contours by area and find the largest valid one
            valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    # Calculate center relative to ROI
                    cx_roi = int(M['m10'] / M['m00'])
                    cy_roi = int(M['m01'] / M['m00'])

                    # Convert to absolute screen coordinates
                    cx_screen = roi_left + cx_roi
                    cy_screen = roi_top + cy_roi
                    return cx_screen, cy_screen
        return None # No target found
    except Exception as e:
        print(f"Error in get_target_position_color: {e}")
        return None

def move_mouse_towards(target_x, target_y, mouse_controller):
    """Moves the mouse a fraction of the distance towards the target."""
    try:
        current_x, current_y = pyautogui.position()

        delta_x = target_x - current_x
        delta_y = target_y - current_y

        # Calculate distance
        distance = math.sqrt(delta_x**2 + delta_y**2)

        # If very close, maybe stop or make tiny adjustments (optional)
        if distance < 1:
            return

        # Calculate movement step size (fraction of remaining distance)
        move_x = delta_x * MOVEMENT_SMOOTHING
        move_y = delta_y * MOVEMENT_SMOOTHING

        # Use integer movement for pynput if needed, or allow float
        mouse_controller.move(move_x, move_y)
    except Exception as e:
        print(f"Error in move_mouse_towards: {e}")

def aim_assist_loop():
    global aim_assist_active, target_lock
    mouse_controller = mouse.Controller()
    sct = mss.mss() # Initialize screen capture object

    while True:
        if aim_assist_active:
            print("Aim assist activated.")
            # Find a target
            current_target = get_target_position_color(sct)

            if current_target:
                print(f"Target found at: {current_target}")
                target_lock = current_target # Lock onto the found target

            # If a target is locked, move towards it
            if target_lock:
                move_mouse_towards(target_lock[0], target_lock[1], mouse_controller)
                time.sleep(AIM_LOOP_DELAY) # Short delay for smooth movement steps
            else:
                # No target found or locked, wait a bit longer before scanning again
                time.sleep(TARGET_SCAN_DELAY)
        else:
            # Wait longer when inactive
            time.sleep(0.1)

# --- Keyboard Listener ---
def on_press(key):
    global aim_assist_active, target_lock
    if key == ACTIVATION_KEY:
        print("Activation key pressed.")
        aim_assist_active = True
        target_lock = None # Reset target lock when key is pressed

def on_release(key):
    global aim_assist_active, target_lock
    if key == ACTIVATION_KEY:
        print("Activation key released.")
        aim_assist_active = False
        target_lock = None # Clear target when key is released

# --- Main Loop ---
if __name__ == "__main__":
    print("Realistic Aim Assist Script (Educational Use Only!)")
    print(f"Hold '{str(ACTIVATION_KEY)}' to activate.")
    print("WARNING: Use in online games may lead to BANS.")
    print("Press Ctrl+C to stop.")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Start the aim assist loop in a separate thread
    aim_assist_thread = threading.Thread(target=aim_assist_loop)
    aim_assist_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nAim Assist Stopped.")
    finally:
        listener.stop()
        aim_assist_thread.join()
        print("Listener and Aim Assist thread stopped.")