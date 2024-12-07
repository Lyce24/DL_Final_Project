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
# parser.add_argument('--embed_path', type=str, required=False, default='bert', help='Path to the ingredient embeddings')
# parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')

# Test Tomorrow:
# multimodal_resnet_gnn_pretrain_da_16_75_25, embedding = gnn
# multimodal_resnet_pretrained_da_16_75_25, embedding = bert
# multimodal_vit_pretrain_da_16_75_25, embedding = bert
# multimodal_resnet_gat_pretrain_da_16_75_25, embedding = gnn_gat
# customized_cnn_da_16_100

# Single Job commands
python ingr_eval.py --model_type customized --model_backbone cnn --model_name customized_cnn_da_16_100 --log_min_max False --batch_size 16

# MultiJob commands
# python ingr_eval.py --model_type baseline --model_backbone cnn_mlp --model_name baseline_cnn_mlp_False_False_True_16_100_norelu --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type bb_lstm --model_backbone vit --model_name bb_lstm_vit_pretrain_da_16_100 --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type bb_lstm --model_backbone resnet --model_name bb_lstm_resnet_pretrain_da_16_100 --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type bb_lstm --model_backbone resnet --model_name bb_lstm_resnet_no_pretrain_da_16_100 --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_pretrain_da_16_100 --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type multimodal --model_backbone resnet --model_name multimodal_resnet_pretrain_da_16_100 --log_min_max False --batch_size 16 &
# python ingr_eval.py --model_type multimodal --model_backbone resnet --model_name multimodal_resnet_no_pretrain_da_16_100 --log_min_max False --batch_size 16 &

# wait

