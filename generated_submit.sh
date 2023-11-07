#!/bin/sh

#BSUB -J myjobname
#BSUB -o /log/myjobname_%J.out
#BSUB -e /log/myjobname_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

echo "Starting Job: myjobname"
echo "Logs will be in the directory: /zhome/95/b/147257/Desktop/deep_learning_project/log"


# Activate the Python virtual environment
source /deep_learning_project/ViT/deeper/bin/activate

# Run the Python script
python /deep_learning_project/ViT/ultralytics_RTDETR.py

