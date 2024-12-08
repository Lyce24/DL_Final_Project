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
# parser.add_argument('--lstm_layers', type=int, required=False, default=1, help='Number of LSTM layers')
# parser.add_argument('--attn_layers', type=int, required=False, default=1, help='Number of Attention layers')

# Finished:
# baseline_model
# bb_lstm_resnet_pretrain_da_16_100
# bb_lstm_vit_pretrain_da_16_100
# multimodal_resnet_ber_pretrained_da_16_75_25, embedding = bert
# multimodal_resnet_gat_v2_pretrained_da_16_75_25, embedding = gat_v2
# multimodal_vit_bert_pretrained_da_16_75_25, embedding = bert
# multimodal_vit_gat_v2_pretrained_da_16_75_25, embedding = gat_v2

# Testing:
# multimodal_resnet_concat_pretrained_da_16_75_25, embedding = concat
# multimodal_resnet_2lstm_bert_pretrained_da_16_75_25, embedding = bert
# smedan_resnet_bert_pretrained_da_16_75_25, embedding = bert
# smedan_resnet_gat_v2_pretrained_da_16_75_25, embedding = gat_v2

# Run later:
# multimodal_resnet_2lstm_concat_pretrained_da_16_75_25, embedding = concat
# multimodal_resnet_2lstm_gat_v2_pretrained_da_16_75_25, embedding = gat_v2

# Define the arguments for the evaluation script
MODEL_TYPE="NutriFusionNet" # "NutriFusionNet", "bb_lstm", "baseline"
MODEL_BACKBONE="resnet" # "resnet", "vit"
MODEL_NAME="multimodal_resnet_concat_pretrained_da_16_75_25"
LOG_MIN_MAX="False"
BATCH_SIZE=16
EMBED_PATH="concat"
LSTM_LAYERS=1
ATTN_LAYERS=1

# Print the job configuration for logging purposes
echo "Running model evaluation with the following configuration:"
echo "Model Type: $MODEL_TYPE"
echo "Model Backbone: $MODEL_BACKBONE"
echo "Model Name: $MODEL_NAME"
echo "Log Min Max: $LOG_MIN_MAX"
echo "Batch Size: $BATCH_SIZE"
echo "Embed Path: $EMBED_PATH"
echo "LSTM Layers: $LSTM_LAYERS"
echo "Attention Layers: $ATTN_LAYERS"

# Run the Python evaluation script with the specified arguments
python ingr_eval.py \
    --model_type $MODEL_TYPE \
    --model_backbone $MODEL_BACKBONE \
    --model_name $MODEL_NAME \
    --log_min_max $LOG_MIN_MAX \
    --batch_size $BATCH_SIZE \
    --embed_path $EMBED_PATH \
    --lstm_layers $LSTM_LAYERS \
    --attn_layers $ATTN_LAYERS

# MultiJob commands
# python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_bert_pretrained_da_16_75_25 --log_min_max False --batch_size 16 --embed_path bert &
# python ingr_eval.py --model_type multimodal --model_backbone vit --model_name multimodal_vit_gat_v2_pretrained_da_16_75_25 --log_min_max False --batch_size 16 --embed_path gat_v2 &

# wait

