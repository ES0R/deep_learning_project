# README

**Authors:** 
- Daniel Jalel (s194291)
- Emil Ramovic (s194149)
- Magnus Madsen (s185382)
- Cato Poulsen (s194127)
- Andrew Blumensen (s194139)

**Date:** December 2023

## Motivation

Our project aims to delve into object detection using various deep-learning techniques, contributing to the very core of the future of self-driving cars. By refining the computer's ability to detect and respond to visual cues, we envision a step closer to truly autonomous, safe, and efficient driving.

As autonomous self-driving cars become more widespread, a notable shift from LiDAR technology to cameras for navigation has occurred. This shift has led to a heightened demand for advanced computer vision systems that can precisely detect and classify traffic-related information. This project compares different deep-learning techniques for object detection in terms of mean average precision (mAP) and real-time analysis/speed analysis.

## Objective

The main objective of this project is to implement and compare the speed and overall performance concerning the accuracy of three distinct object detection models:

- **Self-Made Model**
- **YOLO (You Only Look Once)**
- **RT-DETR**

The self-made model will serve as a simple baseline. These models will be assessed in the context of vehicle applications to gauge their appropriateness for real-time object recognition tasks. Object detection will be carried out on still image datasets as well as image sequence data sets.

## Project File Structure

```
.
Project_Root/
│
├── main/
│   └── train.py
│
├── NOTEBOOKS/
│   └── notebook.ipynb
│
├── main_self.py
├── preprocessing.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```


## Virtual Environment
The project used python `3.10.12`. For testing reasons please use a virtual environment with the `requirements.txt` file.
Preferable either with the name `deep` or `venv` as the `.gitignore` filters it out. To use the virtual environment on windows use the command to generate the virtual environment.

```
python3 -m venv deeper
```
To activate the environment do:

```
source \deeper\Scripts\activate
```
To install the dependencies for the environment use:
```
pip3 install -r requirements.txt
```
and to stop the virtual environment.
```
deactivate
```
Note that if you want cuda to work with the software then use the following commands after installing `requirements.txt`:

## HPC
This makes the terminal more readable and enables the GPU node where the code will be executed.
### GPU Node
```
sxm2sh
```
### Colorful prompt with Git branch
```
PS1='\[\e[1;36m\]\u@\h:\[\e[1;94m\]\w\[\e[0m\] $ '
```

### Modified LS_COLORS
```
export LS_COLORS="di=1;94:*.tar=1;31:*.tgz=1;31:*.arc=1;31:*.rar=1;31:*.zip=1;31:*.gz=1;31:*.bz2=1;31:*.xz=1;31:*.exe=$
```
### Enable colorized output for ls
```
alias ls='ls --color=auto'
```
### Git prompt
```
source ~/.git-prompt.sh
```
### Custom Alias
```
alias train='/zhome/95/b/147257/Desktop/deep_learning_project/train.sh'
```

### Clear
```
clear
```

## Training

Run the `train.sh` script with argument `YOLO` or `RTDETR` for training either. To change training, adjust the `config.json` file. Example:

```
./train.sh YOLO
```
This code loads python `3.10.12`, activates the virtual environment automatically and trains the model.

To run and train the self-made model (SMM) the data preprocessing script `preprocessing.ipynb` must be run first.
After this, the path strings inside `main_self.py` must be changed to the correct output path of `preprocessing.ipynb`.
Now run `main_self.py` from the root path.

## Running Inference and Processing Video

The notebook `notebook.ipynb` loads the best models (self-made model, YOLO and RT-DETR) runs inference, processes a video and calculates FPS.  