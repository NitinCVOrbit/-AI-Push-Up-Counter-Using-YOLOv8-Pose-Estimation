# ------------------------- Import Libraries -------------------------
import cv2                 
import os                   
import random               
from ultralytics import YOLO 
from utils import (          
    align_points_to_fixed_reference_line,
    getAngle,
    draw_pts,
    draw_connection, 
    draw_pushup_bar,
    draw_counter
)

# ------------------------- Load Background Image -------------------------
img_bg_path = 'bg.png'       # Path to background menu image
img_bg = cv2.imread(img_bg_path)  # Read background image
cv2.imshow("PushUp Counter", img_bg)  # Display menu background

# ------------------------- Load YOLO Model -------------------------
model = YOLO('yolov8n-pose.pt')     # Load pretrained YOLOv8 pose model
print("Model device:", model.device)  # Print whether using CPU or GPU (cuda)

# ------------------------- Global Variables -------------------------
counter = 0   # Push-up counter (repetition count)
stage = 'up'  # Current stage of push-up ('up' or 'down')

# # ------------------------- Helper Function: Process One Side -------------------------
def process_side(frame, p, side_indices, relative_indices, connection_body, connection_relative,
                 anchor_idx, counter, stage, fixed_start, fixed_length, mode, pt1, pt2):
    
    # Extract side keypoints
    side_points = [p[i] for i in side_indices]
    relative_points = [p[i] for i in relative_indices]

    # Draw points and body connections
    frame = draw_pts(frame, side_points)
    frame = draw_connection(frame, side_points, connection_body)

    # Calculate joint angle (shoulder-elbow-wrist)
    frame, angle_deg = getAngle(frame, p[anchor_idx], p[anchor_idx + 2], p[anchor_idx + 4])

    # Align body to fixed reference line (normalization)
    aligned_pts = align_points_to_fixed_reference_line(
        relative_points, pt1, pt2, frame, mode,
        fixed_start, fixed_length=fixed_length
    )

    # # Draw aligned points and connections
    frame = draw_pts(frame, aligned_pts[2:])
    frame = draw_connection(frame, aligned_pts, connection_relative)

    # # Draw push-up progress bar and counter
    frame = draw_pushup_bar(frame, angle_deg)
    frame, counter, stage = draw_counter(frame, angle_deg, counter, stage, aligned_pts)

    return frame, counter, stage  # Return updated frame, counter, and stage


# # ------------------------- Helper Function: Pick Random Video -------------------------
def get_random_video(folder_name):
    folder_path = os.path.join('Videos', folder_name)   # Folder path based on side (L/R/F)
    video_files = os.listdir(folder_path)               # List all video files
    selected_video = random.choice(video_files)         # Randomly pick one
    video_path = os.path.join(folder_path, selected_video) # Get full path
    return video_path


# ------------------------- Main Pose Detection Function -------------------------
def run_pose_detection(video_path, side):
    global counter, stage
    cap = cv2.VideoCapture(video_path)   # Open video
    # cap = cv2.VideoCapture("http://10.229.113.206:8080/video")   # Open video

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")   # Error if video not found
        return

    # Define skeleton connections (body parts to draw lines between)
    connection_body = [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 6)]
    connection_relative = [(2, 3), (3, 4), (4, 5), (2, 5)]

    # ---------------- Frame Loop ----------------
    while cap.isOpened():
        ret, frame = cap.read()    # Read next video frame
        if not ret:                # End of video
            counter = 0            # Reset counter
            break

        frame = cv2.resize(frame, (1280, 720))   # Resize frame
        results = model.predict(frame, conf=0.5, verbose=False)  # Run YOLOv8 pose detection
        keypoints = results[0].keypoints.xy[0]   # Extract keypoints
        p = keypoints.cpu().numpy()      # Convert to numpy
        # frame = draw_pts(frame,p)
        print(p)

        # Process LEFT side (or FRONT view includes left)
        if side in ['L', 'F']:
            print("First ==> ", side)
            pt1 = p[5]
            pt2 = p[15]
            frame, counter, stage = process_side(
                frame, p,
                side_indices=[0, 5, 7, 9, 11, 13, 15],   # Nose, Left Shoulder, Left Elbow, etc.
                relative_indices=[0, 7, 5, 11, 13, 15], # Relative body parts for alignment
                connection_body=connection_body,
                connection_relative=connection_relative,
                anchor_idx=5,                      # Anchor at shoulder, end at wrist
                counter=counter, 
                stage=stage,                       # No rotation for left side
                fixed_start=(690, 50),                  # Fixed starting position
                fixed_length=400,                       # Standardized length
                mode=side,
                pt1 = pt1, 
                pt2 = pt2,
            )

        # Process RIGHT side (or FRONT view includes right)
        if side in ['R', 'F']:
            print("Second ==> ", side)
            pt1 = p[16]
            pt2 = p[6]
            frame, counter, stage = process_side(
                frame, p, 
                side_indices=[0, 6, 8, 10, 12, 14, 16], # Nose, Right Shoulder, Right Elbow, etc.
                relative_indices=[0, 8, 6, 12, 14, 16], # Relative body parts for alignment
                connection_body=connection_body,
                connection_relative=connection_relative,
                anchor_idx=6,         # Anchor at shoulder, end at wrist
                counter=counter, stage=stage,                # Rotate for right side alignment
                fixed_start=(190, 50),
                fixed_length=400,
                mode=side,
                pt1 = pt1, 
                pt2 = pt2,
            )

        # Show live pose detection window
        cv2.imshow("PushUp Counter", frame)

        # Break if ESC or 's' key pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('s') or key == ord('S'):
            counter = 0
            break 


# ------------------------- Main Menu Loop -------------------------
while True:
    cv2.imshow("PushUp Counter", img_bg)  # Show background menu
    key = cv2.waitKey(0)                  # Wait for user key press

    # LEFT video
    if key == ord('L') or key == ord('l'):
        print(key)
        try:
            video_path = get_random_video('L')   # Pick random Left video
            print(f"[L] Playing: {video_path}")
            run_pose_detection(video_path, side='L')
        except Exception as e:
            print(e) 

    # RIGHT video
    elif key == ord('R') or key == ord('r'):
        print(key)
        try:
            video_path = get_random_video('R')   # Pick random Right video
            print(f"[R] Playing: {video_path}")
            run_pose_detection(video_path, side='R')
        except Exception as e:
            print(e)

    # FRONT video
    elif key == ord('F') or key == ord('f'):
        print(key)
        try:
            video_path = get_random_video('F')   # Pick random Front video
            print(f"[F] Playing: {video_path}")
            run_pose_detection(video_path, side='F')
        except Exception as e:
            print(e)              

    # Exit program on ESC or 's'
    elif key == 27 or key == ord('s') or key == ord('S'):
        print("Exiting...")
        break
    
    else:
        print("Press L or l for Left Side Detection.")
        print("Press F or f for Front Side Detection.")
        print("Press R or r for Right Side Detection.")
        print("Press S or s for Exit.")

cv2.destroyAllWindows()   # Close all OpenCV windows

# key = cv2.waitKey(0)

