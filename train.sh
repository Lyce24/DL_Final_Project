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
#SBATCH --time=12:00:00
#SBATCH --mem=6G                       # Request 6GB of memory

# Specify a job name:
#SBATCH -J MyPythonJob

# Specify an output file. %j is a special variable that is replaced by the JobID when the job starts.
#SBATCH -o ./output/MyPythonJob-%j.out
#SBATCH -e ./output/MyPythonJob-%j.err

# --- End of Slurm commands ----
# Run the Python script, restricting which GPU to use using CUDA_VISIBLE_DEVICES
# parser.add_argument('--model', type=str, default='inceptionv3', help='Model to train (inceptionv3, convlstm, vit)')
# parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
# parser.add_argument('--log_min_max', type=bool, default=True, help='Use log min max normalization')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
# parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')

python train.py --model inceptionv3 --pretrained False --log_min_max False --batch_size 16 --epochs 50 --save_name inceptionv3_log_regular
