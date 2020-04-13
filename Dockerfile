FROM nvidia/cuda:10.0-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	build-essential \
	libopencv-dev \
	autoconf \
	automake \
	libtool \
	curl \
	vim \
	ffmpeg \
	git \
	python3-pip \
    wget \
    perl

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /gdrive
RUN git clone https://github.com/circulosmeos/gdown.pl.git
RUN chmod +x gdown.pl/gdown.pl
WORKDIR /app/face_seg/checkpoints
RUN /gdrive/gdown.pl/gdown.pl https://drive.google.com/file/d/1Zj_OqQD3huwASMEUfrloKrm7ZYj_DJR6/view?usp=sharing model.pt
WORKDIR /app/yolo_v3
RUN /gdrive/gdown.pl/gdown.pl https://drive.google.com/open?id=1SGjL_7t4FDaPxMM45lcP6o1-jLthHZRv yolov3.weights \
    && /gdrive/gdown.pl/gdown.pl https://drive.google.com/file/d/1e7mgEqI0sJ7jBL3z8R2jWRlnaW_p5y2a/view?usp=sharing tiny_yolov3.weights
WORKDIR /app/face_det/models
RUN /gdrive/gdown.pl/gdown.pl https://drive.google.com/open?id=18qIxdW7zh7zPu7fZKzfglMvq0t2qm6YN Mb_Tiny_RFB_FD_train_input_320.pth

WORKDIR /app
COPY ./face_seg /app/face_seg
COPY ./yolo_v3 /app/yolo_v3
COPY ./face_det /app/face_det
COPY ./sample /app/sample
COPY ./utils /app/utils
COPY ./standard_logs /app/standard_logs

WORKDIR /app
COPY ./main.py /app/main.py

ENV QT_X11_NO_MITSHM 1 