#!/bin/bash
# This is an example batch script for Slurm on Hydra.
#
# The commands for Slurm start with #SBATCH.
# All Slurm commands need to come before the program you want to run.
# In this example, we're running a Python script.
#
# To submit this script to Slurm, use:
# sbatch batch_script.sh
#
# Once the job starts, you will see a file MyPythonJob-****.out.
# The **** will be the Slurm JobID.

# --- Start of Slurm commands -----------

# Set the partition to run on the GPUs partition. The Hydra cluster has the following partitions: compute, gpus, debug, tstaff
#SBATCH --partition=gpus

# Request 1 GPU resource
#SBATCH --gres=gpu:1

# Request an hour of runtime. Default runtime on the compute partition is 1 hour.
#SBATCH --time=6:00:00
#SBATCH --mem=6G                       # Request 6GB of memory

# Specify a job name:
#SBATCH -J MyPythonJob

# Specify an output file. %j is a special variable that is replaced by the JobID when the job starts.
#SBATCH -o ./output/MyPythonJob-%j.out
#SBATCH -e ./output/MyPythonJob-%j.err

# --- End of Slurm commands ----
# Run the Python script, restricting which GPU to use using CUDA_VISIBLE_DEVICES

python train.py --model vit
