# Develop by @HarshJainOfficial

import cv2    
import numpy as np

def region_of_interest(image, vertices):    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

# Process_Image function 
def process_image(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = np.zeros_like(image)

    if lines is not None:
        draw_lines(line_image, lines)

    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

cap = cv2.VideoCapture('test2.mp4')  # Replace 'your_video.mp4' with your video file
while cap.isOpened():
    _, frame = cap.read()
    processed_frame = process_image(frame)
    cv2.imshow('Lane Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
