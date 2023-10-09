import cv2
import numpy as np

cap = cv2.VideoCapture("Coin10.mp4")

if not cap.isOpened():
    print('ไม่สามารถโหลดวิดีโอได้')
else:
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'ขนาดของเฟรม: {frame_width}x{frame_height}')

while True:
    ret, frame = cap.read()
    if not ret:
        print('เลิกอ่านเฟรม')
        break
    else:
        # บริเวณของเฟรมที่จะตรวจจับ กว้าง*สูง
        roi = frame[:1920, :1920]

        # ปรับสีภาพเป็น grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # ลดนอยส์
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # แยกวัตถุออกจากพื้นหลัง
        thresh = cv2.adaptiveThreshold(gray_blur, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11   , 3)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

        result_img = closing.copy()
        contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000 or area > 55000:
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
            counter += 1

        cv2.putText(roi, str(counter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('coindetection', roi)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()