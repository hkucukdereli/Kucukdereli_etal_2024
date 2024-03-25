#!/mnt/colab/colab_shared/anaconda3/bin/python
import torch.nn as nn
import torch.utils.data as data

import argparse
import socket
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
    wandb_track = True
except: 
    wandb_track = False
    print('wandb not installed')



print(f"hostname: {socket.gethostname()}",flush=True)

# where to save the models
output_paths = 'tmp'
wandb_username = "?"
image_size = (512, 640)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--validation_dataset', type=str)
    
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=7, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of class.')
    parser.add_argument('--lr_scheduler', type=str, default='exp', help='exp/plateau')
    parser.add_argument('--model_name', type=str, default='resnet', help='[resnet, alexnet, vgg, squeezenet, densenet, inception, beitBasePatch16]')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    return parser.parse_args()




class ImageLoader(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, manifest, transform):

        self.manifest = manifest
        self.transform = transform

        #Load manifest data, maybe with pandas
        self.dataset = pd.read_csv(manifest)#$,names=['path','label'])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        image = Image.open(sample.path)
        label = sample.label

        if self.transform is not None:
            image = self.transform(image)
        image = image.float()
        
        return image, label


def initialize_model(model_name, input_dims, num_classes, feature_extract, dropout, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    if dropout>0 and model_name!='resnet':
        raise Exception('Dropout is only implemented for resnet!')
    

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained,dropout=dropout)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        # this is how we added dimentions
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
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
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


# args = None

def run_training():
    # global args
    args = parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, input_size, image_mean, image_std = initialize_model(args.model_name, 
                                                                3,
                                                                args.num_class, 
                                                                None, 
                                                                args.dropout, 
                                                                use_pretrained=True)
    model = model.to(device).float()

    ##
    ## Need to verify that indeed the image is 0-1.. if it is 0-255 the normalziation will not have effect!!!
    ##

    ## Train Dataset ##
    ###################

    data_transforms = transforms.Compose([
                                        transforms.Resize((input_size, input_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([
                                                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
                                            ], p=0.7),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize(mean=image_mean,
                                                                std=image_std)
                                        ])
                        
    
    train_dataset = ImageLoader(args.train_dataset, transform = data_transforms)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True, 
                                               pin_memory = True)


    ## Validation Dataset ##
    ########################

    data_transforms_val = transforms.Compose([transforms.Resize((input_size, input_size)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(mean=image_mean,
                                                                std=image_std)])      
                    
    val_dataset = ImageLoader(args.validation_dataset, 
                                transform = data_transforms_val)

    
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


    # Set loggings
    wb_name = f'{args.model_name} do{args.dropout}'
    if wandb_track: wandb.init(project=f"finetunning", entity="oren",name=wb_name)

    if wandb_track: wandb.config.update(args)
    print(train_dataset.dataset['label'].value_counts())
    print(train_dataset.dataset['label'].value_counts())
    if wandb_track: wandb.config.train_data =  len(train_dataset)
    if wandb_track: wandb.config.test_data =  len(train_dataset)


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

        if wandb_track: wandb.log(metrics)

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

            if wandb_track: wandb.log({**metrics, **val_metrics})
            
            if val_metrics["val/acc"] >= best_acc:
                
                #datast_text = f'{args.data_structure}' if args.data_structure!='image' else ''
                save_path = Path(f'{output_paths}','checkpoints','do{args.dropout}')
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'args': args},
                            save_path/ f"model{args.model_name}_{epoch}_acc{acc:.5}.pth")
                tqdm.write('Model saved.')
                #saving training/testing dataframe
                
     
        
if __name__ == "__main__":                    
    run_training()

