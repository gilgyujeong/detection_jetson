import time
from collections import deque
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
pTime = 0
fps_list = deque(maxlen=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV: BRG, mediapipe: RGB

    result = face_detection.process(rgb_frame)

    if result.detections:
        for detection in result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # 얼굴 영역 추출
            face_region = frame[y:y+h, x:x+w]

            # 모자이크 처리
            mosaic_level = 15 # 모자이크 크기 설정
            face_region = cv2.resize(face_region, (mosaic_level, mosaic_level), interpolation=cv2.INTER_LINEAR)
            face_region = cv2.resize(face_region, (w, h), interpolation=cv2.INTER_NEAREST)

            # 모자이크된 얼굴 다시 삽입
            frame[y:y+h, x:x+w] = face_region

            # 얼굴 주위에 경계 상자 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime
    fps_list.append(fps)
    avg_fps = sum(fps_list) / len(fps_list)

    cv2.putText(frame, f'FPS: {int(avg_fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("face Mosaic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()