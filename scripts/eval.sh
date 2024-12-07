#!/bin/bash

# Slurm commands
#SBATCH --partition=gpus             # Use the GPU partition
#SBATCH --gres=gpu:1                 # Request 3 GPUs
#SBATCH --time=01:00:00              # Maximum runtime
#SBATCH --mem=6G                     # Memory allocation
#SBATCH -J Eval                      # Job name
#SBATCH -o ./output/eval-%j.out  # Standard output
#SBATCH -e ./output/eval-%j.err  # Standard error

# parser.add_argument('--model_type', type=str, required=True, help='Type of the model')
# parser.add_argument('--model_backbone', type=str, required= True, help='Model to eval')
# parser.add_argument('--model_name', type=str, required=True, help='Name of the model checkpoint to save')
# parser.add_argument('--log_min_max', type=str2bool, required= False, default=False, help='Used log min-max values')
# parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size for evaluation')
# parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')

# python dp_eval.py --model_backbone convlstm --model_name convlstm_lmm_da_v2 --log_min_max True
python ingr_eval.py --model_type baseline --model_backbone cnn_mlp --model_name baseline_cnn_mlp_False_False_True_16_100_norelu --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type bb_lstm --model_backbone vit --model_name bb_lstm_vit_pretrain_da_16_100 --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type bb_lstm --model_backbone resnet --model_name bb_lstm_resnet_pretrain_da_16_100 --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type bb_lstm --model_backbone resnet --model_name bb_lstm_resnet_no_pretrain_da_16_100 --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_pretrain_da_16_100 --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type multimodal --model_backbone resnet --model_name multimodal_resnet_pretrain_da_16_100 --log_min_max False --batch_size 16 &
python ingr_eval.py --model_type multimodal --model_backbone resnet --model_name multimodal_resnet_no_pretrain_da_16_100 --log_min_max False --batch_size 16 &

wait

