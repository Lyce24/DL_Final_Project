#!/bin/bash

# Slurm commands
#SBATCH --partition=gpus             # Use the GPU partition
#SBATCH --gres=gpu:1                 # Request No. of GPUs
#SBATCH --nodelist=gpu1705           # Explicitly request GPUs
#SBATCH --time=24:00:00              # Maximum runtime
#SBATCH --mem=24G                    # Memory allocation
#SBATCH -J multimodal                      # Job name
#SBATCH -o ./output/multimodal-%j.out      # Standard output
#SBATCH -e ./output/multimodal-%j.err      # Standard error

# parser.add_argument('--model_type', type=str, required= True, help='Model Type (multimodal, bb_lstm, baseline)')
# parser.add_argument('--model_backbone', type=str, required= True, help='Model Backbone (convlstm, vit, clv2, convnx)')
# parser.add_argument('--pretrained', type=str2bool, default=False, help='Use pre-trained weights')
# parser.add_argument('--log_min_max', type=str2bool, default=False, help='Use log min max normalization')
# parser.add_argument('--da', type=str2bool, default=True, help='Use data augmentation')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
# parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  

# v5 - 16 batch size, 100 epochs, no l2, no log min max, data augmentation, no patience, no patience, hidden_dim_2 = 1024
# v6 - Backbone + LSTM, 16 batch size, 100 epochs, no l2, no log min max, data augmentation, no patience, no patience, hidden_dim_1 = 1024, hidden_dim_2 = 512
# v7 - Backbone+ Bidirectional LSTM + Positional Encoding

# Run jobs in parallel, assigning specific GPUs using CUDA_VISIBLE_DEVICES
# Define the arguments for the training script
MODEL_TYPE="multimodal"
MODEL_BACKBONE="resnet"
PRETRAINED="False"
LOG_MIN_MAX="False"
DATA_AUGMENTATION="True"
BATCH_SIZE=16
EPOCHS=100
PATIENCE=100
# save_name = model_type + "_" + model_backbone + "_" + pretrained + "_" + log_min_max + "_" + da + "_" + batch_size + "_" + epochs
SAVE_NAME="multimodal_resnet_no_pretrain_da_16_100"

# Print the job configuration for logging purposes
echo "Running model training with the following configuration:"
echo "Model Type: $MODEL_TYPE"
echo "Model Backbone: $MODEL_BACKBONE"
echo "Pretrained: $PRETRAINED"
echo "Log Min Max: $LOG_MIN_MAX"
echo "Data Augmentation: $DATA_AUGMENTATION"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Save Name: $SAVE_NAME"

# Run the Python training script with the specified arguments
python ingr_train.py \
    --model_type $MODEL_TYPE \
    --model_backbone $MODEL_BACKBONE \
    --pretrained $PRETRAINED \
    --log_min_max $LOG_MIN_MAX \
    --da $DATA_AUGMENTATION \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --save_name $SAVE_NAME