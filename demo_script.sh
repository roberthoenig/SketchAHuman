#!/bin/bash

# Run the python script in the background
python3 interface.py &

# Store the PID of the python script in a variable
PID=$!

# Wait for the python script to finish executing
wait $PID

# Copy the directory to remote server
ssh fpaesde@euler.ethz.ch "rm -r SketchAHuman/submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/doodle_images/"
scp -r submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/doodle_images/ fpaesde@euler.ethz.ch:SketchAHuman/submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/

# Connect to remote server via ssh and run the command, store the job ID
job_id=$(ssh fpaesde@euler.ethz.ch "sbatch -G 1 -n 1 --gres=gpumem:20G --time=20 --mem-per-cpu=16G --wrap='python run.py ShapeModel_test'")

# Extract the job ID from the ssh command output using awk
job_id=$(echo $job_id | awk '{print $4}')

# Wait for the job to complete
ssh fpaesde@euler.ethz.ch "squeue -j $job_id" | awk '{if(NR>1)print $5}' | grep -v "R" > /dev/null
while [ $? -ne 0 ]
do
  sleep 120
  ssh fpaesde@euler.ethz.ch "squeue -j $job_id" | awk '{if(NR>1)print $5}' | grep -v "R" > /dev/null
done

# Copy the directory back to local machine
LAST_DIR=$(ssh fpaesde@euler.ethz.ch "ls -d SketchAHuman/experiments/ShapeModel_test/*/ | tail -n 1")
scp -r fpaesde@euler.ethz.ch:$LAST_DIR experiments/ShapeModel_test/

#Open in meshlab
LAST_LOCAL_DIR=$(ls -d experiments/ShapeModel_test/*/ | sort | tail -n 1)
meshlab $(ls -t $LAST_LOCAL_DIR/*.png.ply | head -1)
