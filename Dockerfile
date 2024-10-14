FROM python:3.12

# 작업 디렉토리 생성 및 설정
WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip install --no-cache-dir opencv-python mediapipe

COPY . /app

# 코드 실행
CMD ["python", "face_mosaic_fps.py"]