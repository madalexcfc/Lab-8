import cv2
import numpy as np

marker_image = cv2.imread("variant-10.jpg", cv2.IMREAD_GRAYSCALE)
_, marker_binary = cv2.threshold(marker_image, 150, 255, cv2.THRESH_BINARY)

orb = cv2.ORB_create()

kp_marker, des_marker = orb.detectAndCompute(marker_binary, None)

cap = cv2.VideoCapture(0)

flip_frame = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)

    kp_frame, des_frame = orb.detectAndCompute(binary_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_marker, des_frame) if des_frame is not None else []

    if len(matches) > 10:
        src_pts = np.float32([kp_marker[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        marker_center = np.mean(dst_pts, axis=0)[0]

        h, w = frame.shape[:2]
        center_square = (w//2 - 75, h//2 - 75, w//2 + 75, h//2 + 75)

        if (center_square[0] < marker_center[0] < center_square[2] and
            center_square[1] < marker_center[1] < center_square[3]):
            flip_frame = not flip_frame

    if flip_frame:
        frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (center_square[0], center_square[1]), 
                  (center_square[2], center_square[3]), (0, 255, 0), 2)

    cv2.imshow("Marker Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
