#!/mnt/colab/colab_shared/anaconda3/bin/python

import os
from pathlib import Path
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from fine_tunning import *

output_paths = 'path_to_save'
image_size = (512, 640)

def parse_args():    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Evaluate a neural network on a batch of data')
    
    # Required positional argument
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--validation_dataset', type=str)
    parser.add_argument('--path', type=str, help='Model path')

    return parser.parse_args()

def loadmodel(model_path):
    model_name = os.path.basename(model_path)
    print(f'Loading model {model_name}')

    model_data = torch.load(f'{model_path}')
    args = model_data['args']
    print(args)

    model, input_size, image_mean, image_std = initialize_model(args.model_name, 
                                                                3,
                                                                args.num_class, 
                                                                None, 
                                                                args.dropout, 
                                                                use_pretrained=True)

    return model, input_size, image_mean, image_std, args

def run_shap():
    # global args
    sysargs = parse_args()
    model_path = sysargs.path
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, model_args = loadmodel(model_path)

    model.load_state_dict(model_data['model_state_dict'])

    model = model.to(device).float()

    ### Transform ###
    data_transforms_val = transforms.Compose([transforms.Resize((input_size, input_size)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(mean=image_mean,
                                                                std=image_std)])   
    
    ### Load the training dataset ###
    train_dataset = ImageLoader(sysargs.train_dataset, transform = data_transforms)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True, 
                                               pin_memory = True)
    
    ### Load the testing dataset ###                    
    test_dataset = ImageLoader(sysargs.validation_dataset, 
                                transform = data_transforms_val)

    
    print(f'Validation set size:{test_dataset.__len__()}')
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model.eval()
    torch.set_grad_enabled(True)

    ### Training ###
    for imgs, targets in tqdm(train_loader, desc=f'Training...'):
        imgs = imgs.to(device)
        e = shap.DeepExplainer(model, imgs)

    ### Testing ###
    for imgs, targets in tqdm(test_loader, desc=f'Testing...'):
        imgs = imgs.to(device)
        shap_values = e.shap_values(imgs)
        
        shap_values = [val.mean(axis=1) for val in shap_values]
        for ij, shap in enumerate(shap_values):
            # Save SHAP 0
            save_path = Path(f'{output_paths}',f'{args.model_name}','shap_0')
            save_path.mkdir(parents=True, exist_ok=True)
            imwrite(save_path/f'shap0_{ij:05}.tif', shap[0].astype('float32'))

            # Save SHAP 1
            save_path = Path(f'{output_paths}',f'{args.model_name}','shap_1')
            save_path.mkdir(parents=True, exist_ok=True)
            imwrite(save_path/f'shap1_{ij:05}.tif', shap[1].astype('float32'))
        
        tqdm.write('SHAP results saved.')

    return None

if __name__=='__main__':
    run_shap()