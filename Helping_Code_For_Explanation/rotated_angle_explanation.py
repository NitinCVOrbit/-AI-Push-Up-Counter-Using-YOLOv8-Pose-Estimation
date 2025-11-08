import cv2
import math
import numpy as np

# ---------------- Draw Line Between Two Points ----------------
def draw_line(img, pt1, pt2, color=(255,255,255)):    
    cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, 4)
    return img


# ---------------- Draw Keypoints ----------------
def draw_pts(img, points,color1 = (0,255,0),color2 = (255,0,255)):
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, color1, -1)   # green filled
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, color2, 2) # magenta border
    return img


# ---------------- Draw Skeleton Connections ----------------
def draw_connection(img, points, connection, color1 = (255,255,255), color2 = (0,0,255)):
    for k, (i,j) in enumerate(connection):
        color = color2 if k == len(connection) - 1 else color1
        img = draw_line(img, points[i], points[j], color)
    return img


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
    fixed_start = None, fixed_y=50,
    fixed_length=400, padding=15
):
    print(mode)
    print(fixed_start)
    h, w, _ = frame.shape

    # --- Step 1: Original line calculation ---
    # Calculate dirction vector (dx, dy) between pt1 and pt2
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1] 

    # cv2.circle(frame,(int(dx),int(dy)),10,(255,255,255),-1)

    # Find original line length using Euclidean distance formula
    orig_len = math.hypot(dx, dy)

    # Find center (midpoint) coordinates of the line pt1→pt2
    center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    
    # cv2.circle(frame,(int(center[0]),int(center[1])),10,(0,255,0),-1)

    # --- Step 2: Rotate points ---
    # Calculate rotation angle to align line horizontally (with extra rotate_angle)
    angle = -math.degrees(math.atan2(dy, dx))

    # cv2.putText(frame, f"{angle}", (int(pt1[0]) - 10, int(pt1[1]) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0))    

    # Rotate all points around the line center
    rot = [rotate_point(p, center, angle) for p in points]

    # Draw aligned points and connections
    # frame = draw_pts(frame, rot, color1=(255,0,0), color2=(0,255,0))
    # frame = draw_connection(frame, rot, connection_relative, color1=(100,100,200), color2=(255,255,0))

    # Rotate pt1 and pt2 themselves for alignment reference
    rot_pt1, rot_pt2 = rotate_point(pt1, center, angle), rotate_point(pt2, center, angle)

    # Recalculate new center after rotation
    rot_center = ((rot_pt1[0] + rot_pt2[0]) / 2, (rot_pt1[1] + rot_pt2[1]) / 2)
    
    # cv2.circle(frame,(int(rot_center[0]),int(rot_center[1])),10,(0,255,0),-1)

    # --- Step 3: Scaling ---
    # Calculate scaling factor to resize line to fixed_length
    scale = fixed_length / orig_len
    # cv2.putText(frame,f'Fixed Length == {fixed_length}',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'Original Length == {orig_len}',(50,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'Scale == {scale}',(50,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    # Scale all rotated points outward/inward relative to rotated center
    scaled = [(rot_center[0] + (x - rot_center[0]) * scale,
               rot_center[1] + (y - rot_center[1]) * scale) for x, y in rot]
    
    #     # Draw aligned points and connections
    # frame = draw_pts(frame, scaled, color1=(255,0,0), color2=(0,255,0))
    # frame = draw_connection(frame, scaled, connection_relative, color1=(100,100,200), color2=(255,255,0))

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
    
    # cv2.putText(frame,f'w/2_X == {int(w/2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'fixed_Y == {int(fixed_y)}',(250,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'rot_center_X == {int(rot_center[0])}',(50,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'rot_center_Y == {int(rot_center[1])}',(250,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'shift_X == {int(shift_x)}',(50,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # cv2.putText(frame,f'shift_Y == {int(shift_y)}',(250,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    # Apply shift to all scaled points
    shifted = [(x + shift_x, y + shift_y) for x, y in scaled]
    # #         # Draw aligned points and connections
    # frame = draw_pts(frame, shifted, color1=(255,0,0), color2=(0,255,0))
    # frame = draw_connection(frame, shifted, connection_relative, color1=(100,100,200), color2=(255,255,0))

    # --- Step 5: Clamping ---
    # Ensure all points stay within frame boundaries (respecting padding margin)
    return [(max(padding, min(w - padding, x)),
             max(padding, min(h - padding, y))) for x, y in shifted] , frame

# Define skeleton connections (body parts to draw lines between)
connection_body = [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 6)]
connection_relative = [(2, 3), (3, 4), (4, 5), (2, 5)]


side_indices=[0, 5, 7, 9, 11, 13, 15]   # Nose, Left Shoulder, Left Elbow, etc.
relative_indices=[0, 7, 5, 11, 13, 15] # Relative body parts for alignment

# ---------------- Example Run ----------------
def main():

    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 25

    p = [[     305.19   ,   338.83],
        [     306.83  ,    322.82],
        [     309.09   ,   331.62],
        [     345.88    ,  308.41],
        [      347.7   ,   319.34],
        [     406.14 ,     350.67],
        [     424.85   ,   342.77],
        [     470.21  ,    459.62],
        [     514.58  ,    434.02],
        [     438.09  ,    572.04],
        [     459.19   ,   538.34],
        [     666.97   ,   380.55],
        [     675.23   ,   402.89],
        [     879.92  ,    461.92],
        [     884.27  ,    452.68],
        [     1094.7    ,  470.98],
        [     1052.3    ,   473.7]]
    
    pt1 = p[5]  # anchor point
    pt2 = p[15]   # end point

    # Extract side keypoints
    side_points = [p[i] for i in side_indices]
    relative_points = [p[i] for i in relative_indices]

    # Draw points and body connections
    # frame = draw_pts(frame, side_points)
    # frame = draw_connection(frame, side_points, connection_body)
    aligned_pts, frame = align_points_to_fixed_reference_line(relative_points, pt1, pt2, frame, mode = 'L')

    # Draw aligned points and connections
    frame = draw_pts(frame, aligned_pts)
    frame = draw_connection(frame, aligned_pts, connection_relative)

    # Display the result
    cv2.imshow("Aligned Points", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

