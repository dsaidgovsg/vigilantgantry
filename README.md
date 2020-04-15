<h2 align="center">VigilantGantry (Face Segmentation AI Engine)</h2>

## About

VigilantGantry is an open source implementation of an AI-driven automated temperature screening gantry that augments existing thermal systems in use, to enhance the rate of contactless screening. The software is designed to address the limitations of current temperature screening systems and staff fatigue in the temperature screening process.

The intention of the project is to provide automated access control and thermal scanning to limit exposure of frontline staff to large numbers of human traffic flow, while still ensuring that users of the gantry are adequately checked.
 
An alert will sound once the system has detected a febrile temperature referencing the thermal imagery from thermal system in use. Such alerts are customizable and should be catered according to ground operations.

VigilantGantry is useful for automating high traffic volume sites screening in detection of symptomatic COVID-19 cases. It aids ground staff to stay vigilant against COVID-19. The solution is also able to interface with facial recognition solutions for contact tracing, if required. On detecting a febrile patient, a secondary check should be conducted using HSA-registered Class B thermometer.  

## Architecture

The architecture is modular and can be integrated with off-the-shelf electronics, thermal systems and optical cameras:

 ![Architecture Diagram](asset/achitecture.png)

 ![Equipment Diagram](asset/equipment.png)


We have tested the following off-the-shelf systems and solutions:

* Omnisense Sentry MKIII (webcam)
* STELOP IFSS (webcam)
 
Here are suggestions for the open-source community to expand the functionalities:

* Integrate with and test on other thermal systems and thermal cameras
* Integrate with and test on other gantries and door access systems
* Integrate with self-declaration apps for contact tracing purposes
* Integrate with facial recognition technology for contact tracing purposes
* Expand the video analytics features such as detection of coughing, running nose, unwell symptoms, mask on/off, etc.
 
## Trial at NUS

<h2 align="center"><a href="https://www.youtube.com/watch?v=4quAADmKs40"><img src="asset/logo.jpg" width="200"></a><h2/">

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


## Contact us
Find out more about GovTech Data Science & Artificial Intelligence Division at our Medium blog.
 