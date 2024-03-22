#!/mnt/colab/colab_shared/anaconda3/bin/python
import os
import glob
from tqdm import tqdm
import argparse
import sys
sys.path.append('/home/oamsalem/code/Projects/Hakan')
from pathlib import Path

from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import pickle
import socket
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from DAN.video_data_parse import get_data_samples 
import h5py
import socket
print(f"hostname: {socket.gethostname()}")
sys.stdout.flush()
import wandb
from torchvision import datasets, models, transforms
import re
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str,choices=['test', 'test_', 'preference'])
    parser.add_argument('--data_split', type=str,default='chunks', choices=['chunks', 'start_end'])
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=7, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of class.')
    parser.add_argument('--lr_scheduler', type=str, default='exp', help='exp/plateau')
    parser.add_argument('--run_debug', action='store_true', help='Run debug mode. in-which I just try to detect running')
    parser.add_argument('--shuffle', action='store_true', help="In this case I'll shuffle the labels, test should be 0.5 accuracy")
    parser.add_argument('--model_name', type=str, default='resnet', help='[resnet, alexnet, vgg, squeezenet, densenet, inception, beitBasePatch16]')
    parser.add_argument('--crop',nargs="+", type=int, default=None, help='top, bottom, left, right image size is (512, 640)')
    parser.add_argument('--data_structure',type=str, default='image', help='image/image3D/diff/diff3D/image3DX/diff3DX, X mark the next/prev frame')
    parser.add_argument('--bkg_mask',type=str, default='no_mask', help='no_mask/DLCmask should we mask the bk of the image')
    parser.add_argument('--force_days',nargs="+", type=int, default=None, help='days to run, e.g 1 2 3, don"t send if not force')

    parser.add_argument('--mice',nargs="+", type=str, default='all', help='mice to run HK90 HK89 HK94 HK99 HK127')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    return parser.parse_args()


run_date = '010722'
# if socket.gethostname() == 'colab02':
#     base_path = '/home/oamsalem/test_data_on_ssd/'
# else:
#     base_path = '/mnt/anastasia/data/behavior/hakan/'

base_path = '/mnt/anastasia/data/behavior/hakan/'
base_path_cache = '/mnt/ssd_cache/manual_cache/'
mouse_orient = pickle.load(open('/mnt/anastasia/data/behavior/hakan/oren/mouse_orient_map.pkl','rb'))
image_size = (512, 640)
toTens = transforms.ToTensor()

class ImageLoader(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, full_df, transform, crop, data_structure, bkg_mask, res_transform, idim: list):

        self.full_df = full_df
        self.transform = transform
        self.crop = crop
        self.data_structure = data_structure
        self.df_len = len(full_df)
        self.bkg_mask = bkg_mask
        self.res_transform = res_transform
        self.idim = idim

        self.max_per_day = {(row['mouse'],row['date']):row['index'] for _,row in full_df.reset_index().groupby(['mouse','date']).max().reset_index().iterrows()} # get the max index per mouse and date


        if self.idim in self.data_structure:
            self.conc_step = 1 if self.data_structure.split(self.idim)[1] == '' else int(self.data_structure.split(self.idim)[-1])

        if self.crop is not None and self.idim in self.data_structure:
            raise NotImplementedError('crop is not implemented for diff3D/image3D')
        if self.crop is not None and bkg_mask != 'no_mask':
            raise NotImplementedError('crop is in full image coordinates, but with mask the image is in 244 coordinates')
        
        if self.bkg_mask == 'DLCmask':
            mouse_date =  full_df.drop_duplicates(subset=['mouse','date'])[['mouse','date']].values
            self.mouse_hdf_fls = {tuple(i):h5py.File(f'/mnt/anastasia/data/behavior/hakan/{i[0]}/{i[1]}_{i[0]}/{i[0]}_{i[1]}_1_background_mask.h5', 'r') for i in mouse_date}


        

    def __len__(self):
        return len(self.full_df)

    def load_image(self, mouse, date, frame_n):
        #
        #
        # I might be able to make this one faster using try/except instead of path.exists
        #
        #Cache image loader
        try:
            image = Image.open(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        except:
            os.makedirs(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs', exist_ok=True)
            image = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
            image.save(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        return image
    
    def load_hakan_crop(self, mouse, date, frame_n):
        for i in range(4):
            try:
                try:
                    image = Image.open(f'{base_path_cache}/{mouse}/{date}_{mouse}/DLCmask/{mouse}_{date}_{frame_n}.jpg')
                except:
                    try:
                        image = np.array(self.load_image(mouse, date, frame_n))
                        mask = self.mouse_hdf_fls[(mouse, date)]['data'][frame_n]
                        image[~mask] = 128
                        image = Image.fromarray(image)
                        image = self.res_transform(image)
                        os.makedirs(f'{base_path_cache}/{mouse}/{date}_{mouse}/DLCmask', exist_ok=True)
                        image.save(f'{base_path_cache}/{mouse}/{date}_{mouse}/DLCmask/{mouse}_{date}_{frame_n}.jpg')
                    except:
                        print(f'{mouse} {date} {frame_n}')
                        #mask = self.mouse_hdf_fls[(mouse, date)]['data'][frame_n]
                        raise Exception(f'Cannot load DLCmask {mouse} {date} {frame_n}')
                return image
            except:
                print(f"Did not manage to load {mouse} {date} {frame_n}, attempt {i}/4")
                time.sleep(0.2)
        raise Exception("FINAL - load_hakan_crop - did not managed to load {mouse} {date} {frame_n} 4 attempts")


    def __getitem__(self, idx):
        sample = self.full_df.iloc[idx]
        mouse, date = sample.mouse, sample.date
        frame_n = sample.name

        # if frame_n>1 and os.path.exists(f'{base_path}/{mouse}/{date}_{mouse}/img/frame_{sample.name+1}.jpg'):
        #     image_0 = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/img/frame_{sample.name-1}.jpg')
        #     image_1 = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/img/frame_{sample.name}.jpg')
        #     image_2 = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/img/frame_{sample.name+1}.jpg')
        #     image = Image.fromarray(np.rollaxis(np.array([np.array(image_0),np.array(image_1),np.array(image_2)]), 0,3))
        # else:
        #image = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_1_{frame_n}.jpg').convert('RGB')
        load_image = self.load_image if self.bkg_mask == 'no_mask' else self.load_hakan_crop

        '''
        try:
            image = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')#.convert('RGB')
        except:
            print(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg failed!')
            raise Exception(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        '''
        image = load_image(mouse, date, frame_n)
        if self.data_structure=='diff':
             image_0 = load_image(mouse, date, max(frame_n-1,1))
             image = ImageChops.subtract(image,image_0)
        elif 'diff3D' in self.data_structure:
            image_0 = load_image(mouse, date, max(frame_n-2*self.conc_step,0))
            image_1 = load_image(mouse, date, max(frame_n-1*self.conc_step,0))
            image_2 = load_image(mouse, date, min(frame_n+1*self.conc_step,self.max_per_day[(mouse,date)]))
            image0 = ImageChops.subtract(image_1,image_0)
            image1 = ImageChops.subtract(image,image_1)
            image2 = ImageChops.subtract(image_2,image)
            image = Image.merge("RGB",(image0,image1,image2))
        elif 'image3D' in self.data_structure: 
            image_0 = load_image(mouse, date, max(frame_n-1*self.conc_step,0))
            image_2 = load_image(mouse, date, min(frame_n+self.conc_step,self.max_per_day[(mouse,date)]))
            image = Image.merge("RGB",(image_0,image,image_2))

        elif f'diff{self.idim}' in self.data_structure:
            if self.conc_step != 1:
                raise Exception('Did not implement conc_step != 1, might work but need testing!')
            D = int(self.idim.split('D')[0])
            imgs = []
            for i in range(D//2-D,D//2+1):
                frame_to_load = min(max(frame_n+i,0),self.max_per_day[(mouse,date)])
                imgs.append(load_image(mouse, date, frame_to_load))
            imgs_diff = []
            for i in range(len(imgs)-1):
                imgs_diff.append(toTens(ImageChops.subtract(imgs[i+1],imgs[i])))
            image = torch.vstack(imgs_diff)
        elif f'image{self.idim}' in self.data_structure:
            if self.conc_step != 1:
                raise Exception('Did not implement conc_step != 1, might work but need testing!')
            D = int(self.idim.split('D')[0])
            imgs = []
            for i in range(D//2-D+1,D//2+1):
                frame_to_load = min(max(frame_n+i,0),self.max_per_day[(mouse,date)])
                imgs.append(toTens(load_image(mouse, date, frame_to_load)))
            image = torch.vstack(imgs)

        # #Cache image loader
        # if os.path.exists(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg'):
        #     image = Image.open(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        # else:
        #     os.makedirs(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs', exist_ok=True)
        #     image = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        #     image.save(f'{base_path_cache}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg')
        #image = Image.open(f'{base_path}/{mouse}/{date}_{mouse}/imgs/{mouse}_{date}_{frame_n}.jpg').convert('RGB')
        if self.crop is not None:
            if mouse_orient[(mouse,date)] == 'left':
                image = image.crop((image_size[1]-self.crop[2],self.crop[1],image_size[1]-self.crop[0],self.crop[3]))
            else:
                image = image.crop((self.crop[0],self.crop[1],self.crop[2],self.crop[3]))
        

        if self.transform is not None:
            image = self.transform(image)
            
        label = sample.stim
        return image, label

def set_parameter_requires_grad(model, feature_extracting):
    pass
    #if feature_extracting:
    #    for param in model.parameters():
    #        param.requires_grad = False


def initialize_model(model_name, input_dims, num_classes, feature_extract, dropout, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    if dropout>0 and model_name!='resnet':
        raise Exception('Dropout is only implemented for resnet!')
    
    if model_name == "beitBasePatch16":
        from transformers import  BeitForImageClassification
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        model_ft = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224',num_labels=num_classes, ignore_mismatched_sizes=True)
        #num_ftrs = model_ft.classifier.in_features
        #model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained,dropout=dropout)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        if input_dims>3:
            inp_layer = torch.nn.Conv2d(input_dims,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                inp_layer.weight[:,input_dims//3:input_dims//3+3] = model_ft.conv1.weight
                inp_layer.weight[:,:input_dims//3] = torch.unsqueeze(model_ft.conv1.weight[:,0],1)
                inp_layer.weight[:,input_dims//3+3:] = torch.unsqueeze(model_ft.conv1.weight[:,2],1)
                
            model_ft.conv1 = inp_layer

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print(f"{model_name} does not exists, exiting...")
        exit()

    return model_ft, input_size, image_mean, image_std


args = None


def subsample_per_mousedate(df, pd_sample_seed):
    mouse_date = list(df.groupby(['mouse','date']).mean().index)

    sampled_data = []
    for mouse,date in mouse_date:
        full_df_md = df.query("mouse==@mouse and date==@date")
        min_frame_len = full_df_md['stim'].value_counts().min()
        sampled_data.append(pd.concat([full_df_md.query("stim==0").sample(min_frame_len,replace=False,random_state=pd_sample_seed).copy(),
                                full_df_md.query("stim==1").sample(min_frame_len,replace=False,random_state=pd_sample_seed*10).copy()]))
    return pd.concat(sampled_data)

def expand_mean_std_dims(X: list, idim: str) -> list :
    D = int(idim.split('D')[0])
    if D>3:    
        st = X[0]
        en  = X[2]
        [X.insert(0,st) for i in range(int((D-3)/2))]
        [X.insert(-1,en) for i in range(int((D-3)/2))]
    return X

import socket
def run_training():
    global args

    pd_sample_seed = 12
    chunk_min_size = 20 # For the case of chunks, what is the minimum size of a chunk?
    subample_per_mousedate = 1
    if args is None:
        args = parse_args()
        if args.data_structure in ['image9D','image11D','image13D','image15D']:
            if args.batch_size == 256:
                if socket.gethostname() == 'colab01':
                    args.batch_size = 130
                if socket.gethostname() == 'colab00':
                    args.batch_size = 200
            #args.workers = 16
        print(args.workers)
    print(args)
    # regexp to see if we have a number of image
    m = re.search('(?:image|diff)(\d*)D', args.data_structure)
    idim = f'{m.group(1)}D' if m else '3D' # data structure dims



    crop_text = f'_crop_{args.crop[0]}_{args.crop[1]}_{args.crop[2]}_{args.crop[3]}' if args.crop else ''
    days_text = ''.join([f'D{i}' for i in args.force_days]) if args.force_days else ''

    wb_name = f'{args.data_set} {args.data_structure} {args.bkg_mask} {args.data_split} {crop_text} {days_text} {args.model_name} {"_".join(args.mice)} do{args.dropout}'
    wandb.init(project=f"finetunning_{run_date}", entity="oren",name=wb_name)
    min_diff = 30 if args.run_debug else 100 # what is the length of each bout
    wandb.config.update(args)

    full_df = get_data_samples([args.data_set],debug=args.run_debug,min_diff=min_diff,data_split=args.data_split,chunk_min_size=chunk_min_size, force_days=args.force_days)

    ### fix training/test data, move to function
    #mice = [f'HK121', 'HK123', 'HK124', 'HK125', 'HK127', 'HK128', 'HK90', 'HK95']
    wandb.config.mice =  args.mice
    
    mice = list(np.sort(list(set(full_df.mouse))))#[f'HK124', 'HK89', 'HK94', 'HK99', 'HK127', 'HK125', 'HK88','HK98']
    if args.mice == ['all']:
        mice = list(set(mice) - set(['HK121'])) # Hakan says its better that we remove this mouse
    else:
        mice = args.mice
    #mice = ['HK90']#, 'HK89', 'HK94', 'HK99', 'HK127']
    print("Running on mice:",mice)
    
    

    #subsample per mouse,date
    if subample_per_mousedate:
        full_df_train = full_df.query(f'train==1 and mouse in @mice').copy()
        full_df_train = subsample_per_mousedate(full_df_train, pd_sample_seed)

        full_df_test = full_df.query(f'train==0 and mouse in @mice').copy()
        full_df_test = subsample_per_mousedate(full_df_test, pd_sample_seed)

    else:
        full_df_train = full_df.query(f'train==1 and mouse in @mice').copy()
        min_frame_len = full_df_train['stim'].value_counts().min()
        full_df_train = pd.concat([full_df_train.query("stim==0").sample(min_frame_len,replace=False,random_state=pd_sample_seed).copy(),
                                    full_df_train.query("stim==1").sample(min_frame_len,replace=False,random_state=pd_sample_seed*10).copy()])
        
        full_df_test = full_df.query(f'train==0 and mouse in @mice').copy()
        min_frame_len = full_df_test['stim'].value_counts().min()
        full_df_test = pd.concat([full_df_test.query("stim==0").sample(min_frame_len,replace=False,random_state=pd_sample_seed).copy(),
                                    full_df_test.query("stim==1").sample(min_frame_len,replace=False,random_state=pd_sample_seed*10).copy()])
    
    print(full_df_train['stim'].value_counts())
    print(full_df_test['stim'].value_counts())
    wandb.config.train_data =  len(full_df_train)
    wandb.config.test_data =  len(full_df_test)
    
    if args.shuffle:
        print("Run shuffle~~~")
        rnd = np.random.RandomState(12)
        full_df_train['stim'] = rnd.permutation(full_df_train['stim'].values)
        full_df_test['stim'] = rnd.permutation(full_df_test['stim'].values)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    model, input_size, image_mean, image_std = initialize_model(args.model_name, int(idim.split('D')[0]) ,args.num_class, None, args.dropout, use_pretrained=True)
    model = model.to(device)

    #expand image_mean and image_std to match the number of channels
    image_mean = expand_mean_std_dims(image_mean, idim)
    image_std = expand_mean_std_dims(image_std, idim)
    print(image_mean)
    print(image_std)

    ##
    ##
    ##
    ## Need to verify that indeed the image is 0-1.. if it is 0-255 the normalziation will not have effect!!!
    ##
    ##
    ##
    ##
    res_transform = transforms.Resize((input_size, input_size))
    no_transform = transforms.Lambda(lambda x: x)
    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor() if idim == '3D' else no_transform,
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)  if idim not in args.data_structure else x),
        transforms.Normalize(mean=image_mean,
                                 std=image_std)
        ])
    
    train_dataset = ImageLoader(full_df_train, transform = data_transforms, crop=args.crop, data_structure=args.data_structure, bkg_mask=args.bkg_mask ,res_transform=res_transform, idim=idim)   # loading dynamically


    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               #sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = True, 
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor() if idim == '3D' else no_transform,
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)  if idim not in args.data_structure else x),
        transforms.Normalize(mean=image_mean,
                                 std=image_std)])      
                                                                      
    val_dataset = ImageLoader(full_df_test, transform = data_transforms_val, crop=args.crop, data_structure=args.data_structure, bkg_mask=args.bkg_mask ,res_transform=res_transform, idim=idim)  # loading dynamically

    
    print(f'Validation set size:{val_dataset.__len__()}')
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)


    criterion_cls = torch.nn.CrossEntropyLoss().to(device)



    optimizer = torch.optim.Adam(model.parameters(),args.lr,betas=(0.9, 0.999), 
                 eps=1e-08, weight_decay=0, amsgrad=False)
    if args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.75)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=4, verbose=True)
    elif args.lr_scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in tqdm(train_loader,desc='Training'):
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out = model(imgs)
            out = out.logits if args.model_name == 'beitBasePatch16' else out

            loss = criterion_cls(out,targets) 

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        metrics = {"train/loss": running_loss, 
                    "train/acc": acc,
                    "train/epoch": epoch+1, 
                    'lr': optimizer.param_groups[0]['lr']}

        wandb.log(metrics)

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in  tqdm(val_loader,desc='Testing'):
        
                imgs = imgs.to(device)
                targets = targets.to(device)
                out = model(imgs)
                out = out.logits if args.model_name == 'beitBasePatch16' else out
                
                loss = criterion_cls(out,targets)

                running_loss += loss.item()
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
            running_loss = running_loss/iter_cnt   

            scheduler.step() if args.lr_scheduler == 'exp' else scheduler.step(running_loss)

            acc = bingo_cnt.float()/float(sample_cnt)
            val_metrics = {"val/loss": running_loss, 
                        "val/acc": np.around(acc.numpy(),4)}

            best_acc = max(val_metrics["val/acc"],best_acc)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, val_metrics["val/acc"], val_metrics['val/loss']))
            tqdm.write("best_acc:" + str(best_acc))

            wandb.log({**metrics, **val_metrics})
            
            if val_metrics["val/acc"] >= best_acc:
                text_add = '_run_debug' if args.run_debug else '' 
                text_add+= '_run_shuffle' if args.shuffle else '' 
                
                if len(mice)==1:
                    mouse_path = mice[0]
                else:
                    mouse_path = f'{len(mice)}_mice'
                
                #datast_text = f'{args.data_structure}' if args.data_structure!='image' else ''
                save_path = Path(f'/mnt/anastasia/data/behavior/hakan/oren/fine_tunning/{run_date}','checkpoints',args.bkg_mask,args.data_structure,args.data_split, args.data_set, crop_text, days_text, mouse_path ,f'do{args.dropout}')
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'args': args},
                            save_path/ f"model{args.model_name}_{epoch}_acc{acc:.5}{text_add}.pth")
                tqdm.write('Model saved.')
                #saving training/testing dataframe
                pickle.dump({'full_df_test':full_df_test,'full_df_train':full_df_train},open(save_path/f"test_train_df{text_add}.pkl",'wb'))

     
        
if __name__ == "__main__":                    
    run_training()





#uFja1fvqk1