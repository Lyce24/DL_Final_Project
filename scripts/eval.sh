#!/bin/bash

# Slurm commands
#SBATCH --partition=gpus             # Use the GPU partition
#SBATCH --gres=gpu:1                 # Request 3 GPUs
#SBATCH --time=01:00:00              # Maximum runtime
#SBATCH --mem=6G                     # Memory allocation
#SBATCH -J Eval                      # Job name
#SBATCH -o ./output/eval-%j.out  # Standard output
#SBATCH -e ./output/eval-%j.err  # Standard error


# python dp_eval.py --model_backbone convlstm --model_name convlstm_lmm_da_v2 --log_min_max True
python ingr_eval.py --model_backbone baseline --model_name baseline --log_min_max False
# python test_eval.py --model_backbone resnet --model_name image_to_mass_no_pretrained_v1 --log_min_max False