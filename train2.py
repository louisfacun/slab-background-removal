import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET2
from tqdm import tqdm
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--checkpoint', type=str, default='best.pth')
parser.add_argument('--resume', type=bool, default=True)
args = parser.parse_args()

# ------- 1. define loss function --------
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCEWithLogitsLoss(size_average=True)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss

# ------- 2. set the directory of training dataset --------
model_name = 'u2net'

model_path = Path('saved_models', model_name)
data_path = Path('data', 'slabs')
train_image_path = data_path / 'train' / 'images'
train_mask_path = data_path / 'train' / 'masks'
val_image_path = data_path / 'val' /'images'
val_mask_path = data_path / 'val' / 'masks'

batch_size_train = args.batch_size
batch_size_val = 1
EPOCHS = args.epochs

# LOAD TRAINING VAL IMAGES AND MASKS PATHS
# print info maybe using logger or print
print("Loading training and validation images and masks...")
print("train_image_path: ", train_image_path)
print("train_mask_path: ", train_mask_path)
print("val_image_path: ", val_image_path)
print("val_mask_path: ", val_mask_path)
train_images = sorted(glob.glob(str(train_image_path / '*.jpg')))
train_masks = sorted(glob.glob(str(train_mask_path / '*.png')))
val_images = sorted(glob.glob(str(val_image_path / '*.jpg')))
val_masks = sorted(glob.glob(str(val_mask_path / '*.png')))

train_set = SalObjDataset(
    img_name_list=train_images,
    lbl_name_list=train_masks,
    transform = transforms.Compose([
        RescaleT(640),
        ToTensorLab(flag=0)
    ])
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_num = len(val_images)
val_set = SalObjDataset(
    img_name_list=val_images,
    lbl_name_list=val_masks,
    transform = transforms.Compose([
        RescaleT(640),
        ToTensorLab(flag=0)
    ])
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size_val,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
print("Loaded training and validation images and masks")
print("Train: ", len(train_images))
print("Val: ", len(val_images))

# ------- 3. define model --------
print("Loading model...")
net = U2NET2(3, 1)
net = nn.DataParallel(net)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(
    net.parameters(), 
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0
)
scaler = torch.cuda.amp.GradScaler(enabled=True)
epoch_start = 0

if args.resume:
    print("loading last model...")
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint) #['model_state_dict']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scaler.load_state_dict(checkpoint['scaler'])
    #epoch_start = checkpoint['epoch']
    #avg_train_loss = checkpoint['avg_train_loss']
    #avg_val_loss = checkpoint['avg_val_loss']
    print("loaded last model")
    #print("epoch: ", epoch_start)
    #print("avg_train_loss: ", avg_train_loss)
    #print("avg_val_loss: ", avg_val_loss)

   
# ------- 5. training process --------
print("---start training...")

best_avg_val_loss = 10000
for epoch in range(epoch_start, EPOCHS):
    running_train_loss = 0
    num_iterations = len(train_loader) 

    start_time = time.time()
    print('EPOCH {}/{}'.format(epoch+1, EPOCHS))

    # TRAINING
    net.train()
    train_loader = tqdm(train_loader, total=num_iterations)
    for i, data in enumerate(train_loader):
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v = Variable(inputs.cuda(), requires_grad=False)
            labels_v = Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v = Variable(inputs, requires_grad=False)
            labels_v = Variable(labels, requires_grad=False)
        
        #optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_train_loss += loss.data.item()
        train_loader.set_postfix({'Loss': loss.data.item()})
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    
    avg_train_loss = running_train_loss / num_iterations
    #running_train_loss = 0
    
    print(f'Epoch: {epoch+1} | Average Training Loss: {avg_train_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')
    
    # VALIDATION
    net.eval()
    running_val_loss = 0
    val_loader = tqdm(val_loader, total=val_num)
    
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v = Variable(inputs.cuda(), requires_grad=False)
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v = Variable(inputs, requires_grad=False)
                labels_v = Variable(labels, requires_grad=False)
            
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            
            running_val_loss += loss.data.item()
            val_loader.set_postfix({'Loss': loss.data.item()})

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

    avg_val_loss = running_val_loss / val_num
    print(f'Epoch: {epoch+1} | Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    if avg_val_loss < best_avg_val_loss:
        best_avg_val_loss = avg_val_loss
        # join model path posix path to str model name and best.pth
        PATH = model_path / (model_name + "_best.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
        }, PATH)
        print("saved best model")

    PATH = model_path / (model_name + "_last.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
    }, PATH)
    print("saved last model")

