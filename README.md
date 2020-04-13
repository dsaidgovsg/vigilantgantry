<h2 align="center">VigilantGantry (Face Segmentation AI Engine)</h2>

## About

[![Alt text](https://img.youtube.com/vi/4quAADmKs40/0.jpg)](https://www.youtube.com/watch?v=4quAADmKs40)

## Prerequisite

* [Docker Engine](https://docs.docker.com/engine/install/binaries/)
* [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

## Docker
```bash
docker build . -t face_seg
docker run --rm --runtime=nvidia -ti -e DISPLAY=$DISPLAY -e  DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1 --name face_seg face_seg:latest
```

## Acknowledgement 
* [Person Detection](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* [Face Detection](Ultra-Light-Fast-Generic-Face-Detector-1MB)
* [Face Segmentation](https://github.com/kampta/face-seg)