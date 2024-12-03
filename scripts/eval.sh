#!/bin/bash

# parser.add_argument('--model_backbone', type=str, required= True, help='Model to eval')
# parser.add_argument('--ingr', type=str2bool, default=False, help='Use ingredient dataset')
# parser.add_argument('--model_name', type=str, required=True, help='Name of the model checkpoint to save')
# parser.add_argument('--log_min_max', type=str2bool, default=True, help='Used log min-max values')
# parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')


# python eval.py --model_backbone inceptionv3 --model_name inceptionv3_lmm_da_v2 --log_min_max True --s inceptionv3_lmm_da_v2
python ingr_eval.py --model_backbone resnet --model_name resnet_ingr_log_da_v3_32_120_30_0.0 --log_min_max False