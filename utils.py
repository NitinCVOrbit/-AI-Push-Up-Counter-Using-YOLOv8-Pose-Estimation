import cv2
import numpy as np
import math


# ---------------- Push-up Progress Bar ----------------
def draw_pushup_bar(frame, angle, min_angle=70, max_angle=180):
    # --- Calculation: Clamp angle between min and max ---
    angle = max(min(angle, max_angle), min_angle)

    # --- Setup: Bar position and size ---
    bar_x, bar_width = 30, 30
    bar_height = 200
    bar_y = (frame.shape[0] - bar_height) // 2

    # --- Setup: Define colors for bar and outline ---
    empty = (230, 230, 230)
    red, green, outline = (0, 0, 255), (0, 255, 0), (50, 50, 50)

    # --- Drawing: Empty background bar ---
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), empty, -1)

    # --- Drawing: Labels for UP and DOWN ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "UP", (bar_x, bar_y - 10), font, 0.6, red, 2)
    cv2.putText(frame, "DOWN", (bar_x - 10, bar_y + bar_height + 25), font, 0.6, green, 2)

    # --- Calculation: Midpoint angle and half bar height ---
    mid_angle = (min_angle + max_angle) // 2
    half_h = bar_height // 2

    # --- Decision + Calculation: Fill bar based on angle ---
    if angle < mid_angle:  # DOWN phase
        # --- Calculation: Ratio + fill height for DOWN (green) ---
        ratio = (mid_angle - angle) / (mid_angle - min_angle)
        fill = int(half_h * ratio)
        # --- Drawing: Fill green bottom half ---
        cv2.rectangle(frame, (bar_x, bar_y + half_h),
                      (bar_x + bar_width, bar_y + half_h + fill), green, -1)
    else:  # UP phase
        # --- Calculation: Ratio + fill height for UP (red) ---
        ratio = (angle - mid_angle) / (max_angle - mid_angle)
        fill = int(half_h * ratio)
        # --- Drawing: Fill red top half ---
        cv2.rectangle(frame, (bar_x, bar_y + half_h - fill),
                      (bar_x + bar_width, bar_y + half_h), red, -1)

    # --- Drawing: Outline around the bar ---
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), outline, 2)

    # --- Return: Modified frame with progress bar ---
    return frame



# ---------------- Rotate Point Around Center ----------------
def rotate_point(pt, center, angle_deg):
    angle_rad = math.radians(angle_deg)
    x, y = pt
    cx, cy = center
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Translate to origin
    x_shifted, y_shifted = x - cx, y - cy

    # Rotate point
    x_rot = x_shifted * cos_a - y_shifted * sin_a
    y_rot = x_shifted * sin_a + y_shifted * cos_a

    # Translate back
    return (x_rot + cx, y_rot + cy)

# ---------------- Align Keypoints to Reference ----------------
def align_points_to_fixed_reference_line(
    points, pt1, pt2, frame,
    mode,
    fixed_start, fixed_y=50,
    fixed_length=600, padding=15
):
    print(mode)
    print(fixed_start)
    h, w, _ = frame.shape

    # --- Step 1: Original line calculation ---
    # Calculate difference vector (dx, dy) between pt1 and pt2
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]

    # Find original line length using Euclidean distance formula
    orig_len = math.hypot(dx, dy)

    # Find center (midpoint) coordinates of the line pt1→pt2
    center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

    # --- Step 2: Rotate points ---
    # Calculate rotation angle to align line horizontally (with extra rotate_angle)
    angle = -math.degrees(math.atan2(dy, dx))

    # Rotate all points around the line center
    rot = [rotate_point(p, center, angle) for p in points]

    # Rotate pt1 and pt2 themselves for alignment reference
    rot_pt1, rot_pt2 = rotate_point(pt1, center, angle), rotate_point(pt2, center, angle)

    # Recalculate new center after rotation
    rot_center = ((rot_pt1[0] + rot_pt2[0]) / 2, (rot_pt1[1] + rot_pt2[1]) / 2)

    # --- Step 3: Scaling ---
    # Calculate scaling factor to resize line to fixed_length
    scale = fixed_length / orig_len

    # Scale all rotated points outward/inward relative to rotated center
    scaled = [(rot_center[0] + (x - rot_center[0]) * scale,
               rot_center[1] + (y - rot_center[1]) * scale) for x, y in rot]

    # --- Step 4: Shifting ---
    if mode == 'F':  
        # Align pt1 exactly to fixed_start
        scaled_pt1 = (rot_center[0] + (rot_pt1[0] - rot_center[0]) * scale,
                      rot_center[1] + (rot_pt1[1] - rot_center[1]) * scale)

        # Calculate shift values so that scaled pt1 moves to fixed_start
        shift_x, shift_y = fixed_start[0] - scaled_pt1[0], fixed_start[1] - scaled_pt1[1]
    else:  
        # Center-align: move rotated center to middle of frame horizontally,
        # and align vertically with fixed_y
        shift_x, shift_y = (w/2 - rot_center[0]), (fixed_y - rot_center[1])

    # Apply shift to all scaled points
    shifted = [(x + shift_x, y + shift_y) for x, y in scaled]

    # --- Step 5: Clamping ---
    # Ensure all points stay within frame boundaries (respecting padding margin)
    return [(max(padding, min(w - padding, x)),
             max(padding, min(h - padding, y))) for x, y in shifted]



# ---------------- Compute Angle Between Three Points ----------------  
def getAngle(img, pt1, pt2, pt3, color=(255, 255, 0), thickness=2, font_scale=0.6):
    # Convert to numpy
    pt1, pt2, pt3 = np.array(pt1), np.array(pt2), np.array(pt3)

    # Vectors
    v1, v2 = pt1 - pt2, pt3 - pt2

    # Normalize and compute angle
    unit_v1, unit_v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_deg = int(round(np.degrees(np.arccos(dot_product))))

    # Draw angle text
    cv2.putText(img, f'{angle_deg} degree',
                (int(pt2[0]) + 13, int(pt2[1])),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img, angle_deg


# ---------------- Draw Line Between Two Points ----------------
def draw_line(img, pt1, pt2, color=(255,255,255)):    
    cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, 4)
    return img


# ---------------- Draw Keypoints ----------------
def draw_pts(img, points):
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, (0,255,0), -1)   # green filled
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (255,0,255), 2) # magenta border
    return img


# ---------------- Draw Skeleton Connections ----------------
def draw_connection(img, points, connection):
    for k, (i,j) in enumerate(connection):
        color = (0,0,255) if k == len(connection) - 1 else (255,255,255)
        img = draw_line(img, points[i], points[j], color)
    return img


# ---------------- Counter Logic (Push-ups) ----------------
def draw_counter(img, angle, counter, stage, align_pts):
    font, color = cv2.FONT_HERSHEY_COMPLEX, (255, 255, 255)

    # ---------------- Counter display settings ----------------
    if counter < 10:
        scale = 2       # font size
        thick = 4       # line thickness
        pos = (20, 65)  # text position (x, y)

    elif counter < 100:
        scale = 2
        thick = 4
        pos = (2, 65)

    else:  # 100 or more
        scale = 1.2
        thick = 3
        pos = (4, 60)

    # ---------------- Push-up detection logic ----------------
    nose_y = align_pts[0][1]   # Y-position of nose
    elbow_y = align_pts[1][1]  # Y-position of elbow

    # Detect going DOWN
    if angle < 95 and elbow_y < nose_y and stage == 'up':
        stage = 'down'

    # Detect going UP and count rep
    elif angle > 130 and stage == 'down':
        counter += 1
        stage = 'up'
    

    # ---------------- Pink quarter circle (top-left corner) ----------------
    center = (0, 0)            # corner of image
    radius = 100               # size of quarter circle
    pink = (255, 0, 255)       # BGR color for pink
    start_angle = 0            # from 0 to 90 degrees
    end_angle = 90
    thickness = -1             # filled

    # Draw the quarter circle
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, pink, thickness)

    # ---------------- Draw counter value ----------------
    cv2.putText(img, str(counter), pos, font, scale, color, thick, cv2.LINE_AA)

    return img, counter, stage

