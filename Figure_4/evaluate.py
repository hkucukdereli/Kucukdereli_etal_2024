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
    parser.add_argument('--eval_dataset', type=str)
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

def evaluate():
    # global args
    sysargs = parse_args()
    model_path = sysargs.path
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, args = loadmodel(model_path)

    model.load_state_dict(model_data['model_state_dict'])

    model = model.to(device).float()

    ### Transform ####
    data_transforms_val = transforms.Compose([transforms.Resize((input_size, input_size)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(mean=image_mean,
                                                                std=image_std)])   

    ### Load dataset to evaluate ###
    eval_dataset = ImageLoader(sysargs.eval_dataset, transform=data_transforms)
    print('Dataset size:', eval_dataset.__len__())
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True, 
                                               pin_memory = True)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model.eval()
    with torch.no_grad():
        outs = []
        for imgs, targets in tqdm(eval_loader, desc=f'Testing {mouse}'):
            imgs = imgs.to(device)
            targets = targets.to(device)
            out = model(imgs)
            out = out.logits if args.model_name=='beitBasePatch16' else out
            # classes = torch.argmax(out, axis=1).cpu().numpy()

            p = nn.functional.softmax(out, dim=1)

            temp_df = pd.DataFrame({'out_0':np.vstack(out.cpu().numpy())[:,0], 
                                    'out_1':np.vstack(out.cpu().numpy())[:,1],
                                    'p_0':np.vstack(p.cpu().numpy())[:,0],
                                    'p_1':np.vstack(p.cpu().numpy())[:,1],})
            outs.append(temp_df)
        
        outs = pd.concat(outs)

        save_path = Path(f'{output_paths}',f'{args.model_name}')
        save_path.mkdir(parents=True, exist_ok=True)
        outs.to_hdf(save_path/'eval_results.h5', 'data')

        print('Results saved.')

    return None

if __name__=='__main__':
    evaluate()