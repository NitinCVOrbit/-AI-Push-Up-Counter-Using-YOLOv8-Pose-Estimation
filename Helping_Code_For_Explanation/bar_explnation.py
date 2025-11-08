
import cv2
import numpy as np

# ---------------- Push-up Progress Bar ----------------
def draw_pushup_bar(frame, angle, min_angle=95, max_angle=130):
    # --- Clamp angle between min and max ---
    angle = max(min(angle, max_angle), min_angle)

    # --- Setup: Bar position and size ---
    bar_x = 30
    bar_width = 30
    bar_height = 200
    bar_y = (frame.shape[0] - bar_height) // 2  # vertically centered

    # --- Colors ---
    empty = (230, 230, 230)
    red, green, outline = (0, 0, 255), (0, 255, 0), (50, 50, 50)

    # --- Draw empty background bar ---
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), empty, -1)

    # --- Draw labels ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "UP", (bar_x, bar_y - 10), font, 0.6, red, 2)
    cv2.putText(frame, "DOWN", (bar_x - 10, bar_y + bar_height + 25), font, 0.6, green, 2)

    # --- Midpoint and half height ---
    mid_angle = (min_angle + max_angle) // 2
    half_h = bar_height // 2

    # --- Fill based on phase ---
    if angle < mid_angle:  # DOWN phase (green fill)
        ratio = (mid_angle - angle) / (mid_angle - min_angle)
        fill = int(half_h * ratio)
        cv2.rectangle(frame, (bar_x, bar_y + half_h),
                      (bar_x + bar_width, bar_y + half_h + fill), green, -1)
    else:  # UP phase (red fill)
        ratio = (angle - mid_angle) / (max_angle - mid_angle)
        fill = int(half_h * ratio)
        print(ratio)
        print(fill)
        cv2.rectangle(frame, (bar_x, bar_y + half_h - fill),
                      (bar_x + bar_width, bar_y + half_h), red, -1)

    # --- Draw bar outline ---
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), outline, 2)

    # --- Return modified frame ---
    return frame


# ---------------- MAIN DEMO ----------------
def main():
    # Create a blank white frame
    frame_height, frame_width = 480, 640
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    angle = 120

    # Draw progress bar
    draw_pushup_bar(frame, angle)

    # Display angle text
    cv2.putText(frame, f"Angle: {int(angle)} deg", (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show window
    cv2.imshow("Push-up Progress Bar Demo", frame)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
