# Project Synopsis for Deep Learning

**Authors:** 
- Daniel Jalel (s194291)
- Emil Ramovic (s194149)
- Magnus Bøje Madsen (s185382)
- Cato Poulsen (s194127)
- Andrew Blumensen (s194139)

**Date:** October 2023

## Motivation

Our project aims to delve into object detection using various deep-learning techniques, contributing to the very core of the future of self-driving cars. By refining the computer's ability to detect and respond to visual cues, we envision a step closer to truly autonomous, safe, and efficient driving.

As autonomous self-driving cars become more widespread, a notable shift from LiDAR technology to cameras for navigation has occurred. This shift has led to a heightened demand for advanced computer vision systems that can precisely detect and classify traffic-related information. This project compares different deep-learning techniques for object detection in terms of mean average precision (mAP) and real-time analysis/speed analysis.

## Objective

The main objective of this project is to implement and compare the speed and overall performance concerning the accuracy of three distinct object detection models:

- **YOLO (You Only Look Once)**
- **ViT (Visual Transformer)**
- **AlexNet**

AlexNet will serve as a simple baseline. These models will be assessed in the context of vehicle and robotics applications to gauge their appropriateness for real-time object recognition tasks. Object detection will be carried out on still image datasets as well as image sequence data sets.

## Project File Structure

```
.
Project_Root/
│
├── YOLO/
│   
│   ├── models/
│   ├── notebooks/
│   └── scripts/
│
├── ViT/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   └── scripts/
│
├── AlexNet/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   └── scripts/
├── data
├── requirements.txt
├── README.md
└── .gitignore
```

In the above structure:
- `data/` contains the datasets required for each model.
- `models/` will store the trained model checkpoints.
- `notebooks/` can be used to store Jupyter notebooks.
- `scripts/` can store various utility scripts or the main code to run the experiments.

The project used python `3.10.11`. For testing reasons please use a virtual environment with the `requirements.txt` file preferable either with the name `deep` or `venv` as the `.gitignore` filters it out. To use the virtual environment on windows use the command
```
.\deep\Scripts\activate.ps1
```

and 

```
deactivate
```

to stop the virtual environment. Note that if you want cuda to work then use the following commands:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```