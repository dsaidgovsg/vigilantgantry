<h2 align="center">VigilantGantry (Face Segmentation AI Engine)</h2>

## About

[![Alt text](/sample/vigilantgantry_image.jpg?s=200)](https://www.youtube.com/watch?v=4quAADmKs40)

VigilantGantry is a project by the Government Technology Agency of Singapore developed in response to the COVID19 situation. VigilantGantry integrates existing thermal camera sensors with a physical gantry, along with other systems such as cameras and backend systems. 

The intention of the project is to provide automated access control and thermal scanning to limit exposure of frontline staff to large numbers of human traffic flow, while still ensuring that users of the gantry are adequately checked.

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