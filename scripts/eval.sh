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
# multimodal_vit_bert_pretrained_da_16_75_25, embedding = bert
# multimodal_vit_gat_v2_pretrained_da_16_75_25, embedding = gat_v2
# multimodal_resnet_concat_pretrained_da_16_75_25, embedding = concat
# smedan_resnet_bert_pretrained_da_16_75_25, embedding = bert
# smedan_resnet_gat_v2_pretrained_da_16_75_25, embedding = gat_v2
# multimodal_resnet_2lstm_bert_pretrained_da_16_75_25, embedding = bert (later)

# Single Job commands
python ingr_eval.py --model_type multimodal --model_backbone resnet --model_name multimodal_resnet_concat_pretrained_da_16_75_25 --log_min_max False --batch_size 16 --embed_path concat

# MultiJob commands
# python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_bert_pretrained_da_16_75_25 --log_min_max False --batch_size 16 --embed_path bert &
# python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_gat_v2_pretrained_da_16_75_25 --log_min_max False --batch_size 16 --embed_path gat_v2 &

# wait

