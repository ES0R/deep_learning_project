#!/bin/bash

# Usage: ./submit_job.sh JOB_NAME SUBFOLDER_NAME

# Assign the first and second argument to variables
JOB_NAME=$1
SUBFOLDER_NAME=$2

# Check the number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 JOB_NAME SUBFOLDER_NAME"
  exit 1
fi

#Log folder
mkdir -p log

# Create the submission script with the provided arguments
cat > generated_submit.sh <<EOF
#!/bin/sh

#BSUB -J $JOB_NAME
#BSUB -o /log/${JOB_NAME}_%J.out
#BSUB -e /log/${JOB_NAME}_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

echo "Starting Job: $JOB_NAME"
echo "Logs will be in the directory: /zhome/95/b/147257/Desktop/deep_learning_project/log"


# Activate the Python virtual environment
source /deep_learning_project/$SUBFOLDER_NAME/deeper/bin/activate

# Run the Python script
python /deep_learning_project/$SUBFOLDER_NAME/ultralytics_RTDETR.py

EOF

# Submit the job
bsub < generated_submit.sh
