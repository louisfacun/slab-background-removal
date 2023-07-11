import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from torch.utils.data import random_split
import numpy as np
import glob
import os
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from data_loader import Augment
from model import U2NET
from model import U2NETP
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP

#bce_loss = nn.BCEWithLogitsLoss(size_average=True)
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
model_name = 'u2net' #'u2netp'
data_dir = os.path.join(os.getcwd(), 'train_data_orig' + os.sep)
tra_image_dir = os.path.join('JUTS', 'images' + os.sep)
tra_label_dir = os.path.join('JUTS', 'masks' + os.sep)
image_ext = '.jpg'
label_ext = '.jpg'
model_dir = os.path.join(
    os.getcwd(), 
    'saved_models', 
    model_name + os.sep
)

batch_size_train = 8
batch_size_val = 1
train_num = 0
val_num = 0

# LOAD TRAINING IMAGES AND MASKS PATHS
tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]
	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]
	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("images: ", len(tra_img_name_list))
print("masks: ", len(tra_lbl_name_list))
print("---")

train_images, val_images, train_masks, val_masks = train_test_split(
    tra_img_name_list,
    tra_lbl_name_list,
    test_size=0.20,
    random_state=100
)
print("---")
print("train_images: ", len(train_images))
print("train_masks: ", len(train_masks))
print("---")
print("val_images: ", len(val_images))
print("val_masks: ", len(val_masks))
print("---")

train_num = len(train_images)

train_set = SalObjDataset(
    img_name_list=train_images,
    lbl_name_list=train_masks,
    transform = transforms.Compose([
        RescaleT(572),
        #Augment(),
        ToTensorLab(flag=0)
    ])
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=2
)

val_num = len(val_images)
val_set = SalObjDataset(
    img_name_list=val_images,
    lbl_name_list=val_masks,
    transform = transforms.Compose([
        RescaleT(572),
        ToTensorLab(flag=0)
    ])
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size_val,
    shuffle=False,
    num_workers=2,
)

# ------- 3. define model --------
resume = True
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
    net = nn.DataParallel(net)#.to(device)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)
if torch.cuda.is_available():
    net.cuda()

resume = True
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

if resume:
    checkpoint = torch.load("best.pth")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scaler.load_state_dict(checkpoint['scaler'])
    epoch_start = checkpoint['epoch']
    #avg_train_loss = checkpoint['avg_train_loss']
    print("loaded last model")
    print("epoch: ", epoch_start)
    #print("avg_train_loss: ", avg_train_loss)

import time    
# ------- 5. training process --------
print("---start training...")


EPOCHS = 100

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
        

        optimizer.zero_grad()
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
    best_avg_val_loss = 10000
    num_iterations = len(val_loader)
    val_loader = tqdm(val_loader, total=num_iterations)

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

    avg_val_loss = running_val_loss / num_iterations
    print(f'Epoch: {epoch+1} | Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    if avg_val_loss < best_avg_val_loss:
        best_avg_val_loss = avg_val_loss
        PATH = model_dir + model_name+"_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
        }, PATH)
        print("saved best model")

    PATH = model_dir + model_name+"_last.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
    }, PATH)
    print("saved last model")

