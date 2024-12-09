#!/bin/bash

# Slurm commands
#SBATCH --partition=gpus             # Use the GPU partition
#SBATCH --gres=gpu:1                 # Request No. of GPUs
#SBATCH --nodelist=gpu1701           # Explicitly request GPUs
#SBATCH --time=24:00:00              # Maximum runtime
#SBATCH --mem=24G                    # Memory allocation
#SBATCH -J NutriFusionNetV2                      # Job name
#SBATCH -o ./output/NutriFusionNetV2-%j.out      # Standard output
#SBATCH -e ./output/NutriFusionNetV2-%j.err      # Standard error

# parser = argparse.ArgumentParser(description='Train a model')
# parser.add_argument('--model_type', type=str, required= True, help='Model Type (multimodal, bb_lstm, baseline)')
# parser.add_argument('--model_backbone', type=str, required= True, help='Model Backbone (convlstm, vit, clv2, convnx)')
# parser.add_argument('--embed_path', type=str, required=False, default='bert', help='Path to the ingredient embeddings')
# parser.add_argument('--pretrained', type=str2bool, default=False, help='Use pre-trained weights')
# parser.add_argument('--log_min_max', type=str2bool, default=False, help='Use log min max normalization')
# parser.add_argument('--da', type=str2bool, default=True, help='Use data augmentation')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
# parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
# parser.add_argument('--lstm_layers', type=int, required=False, default=1, help='Number of LSTM layers')
# parser.add_argument('--attn_layers', type=int, required=False, default=1, help='Number of Attention layers')
# parser.add_argument('--fusion', type=str, required=False, default='concat', help='Fusion method for multimodal model')
# parser.add_argument('--task_heads', type=int, required=False, default=1, help='Number of task heads for multimodal model')

# Define the arguments for the training script
# MODEL_TYPE="bb_lstm", "baseline", "NutriFusionNet"
MODEL_TYPE="NutriFusionNet"
MODEL_BACKBONE="vit"
EMBED_PATH="bert"
PRETRAINED="True"
LOG_MIN_MAX="False"
DATA_AUGMENTATION="True"
BATCH_SIZE=16
EPOCHS=75
PATIENCE=25
LSTM_LAYERS=2
ATTN_LAYERS=2

# save_name = model_type + "_" + model_backbone + "_" + embed_path + "_" + pretrained + "_" + log_min_max + "_" + da + "_" + batch_size + "_" + epochs + "_" + patience
SAVE_NAME="NutriFusionNet_vit_2lstm_2attn_bert_pretrained_da_16_75_25"

# Print the job configuration for logging purposes
echo "Running model training with the following configuration:"
echo "Model Type: $MODEL_TYPE"
echo "Model Backbone: $MODEL_BACKBONE"
echo "Embed Path: $EMBED_PATH"
echo "Pretrained: $PRETRAINED"
echo "Log Min Max: $LOG_MIN_MAX"
echo "Data Augmentation: $DATA_AUGMENTATION"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Save Name: $SAVE_NAME"
echo "LSTM Layers: $LSTM_LAYERS"
echo "Attention Layers: $ATTN_LAYERS"

# Run the Python training script with the specified arguments
python test_train.py \
    --model_type $MODEL_TYPE \
    --model_backbone $MODEL_BACKBONE \
    --embed_path $EMBED_PATH \
    --pretrained $PRETRAINED \
    --log_min_max $LOG_MIN_MAX \
    --da $DATA_AUGMENTATION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --save_name $SAVE_NAME \
    --lstm_layers $LSTM_LAYERS \
    --attn_layers $ATTN_LAYERS 