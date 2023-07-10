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

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
import numpy as np

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)

def multi_iou(d0, d1, d2, d3, d4, d5, d6, labels_v):
    pass

def make_binary(d):
    array = d.numpy()
    binary = np.where(array > 0.6, 1,0).astype('float32')
    binary = binary.reshape((1, 1, 572, 572))
    tensor = torch.from_numpy(binary)
    return tensor

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = bce_loss(make_binary(d0),labels_v)
	loss1 = bce_loss(make_binary(d1),labels_v)
	loss2 = bce_loss(make_binary(d2),labels_v)
	loss3 = bce_loss(make_binary(d3),labels_v)
	loss4 = bce_loss(make_binary(d4),labels_v)
	loss5 = bce_loss(make_binary(d5),labels_v)
	loss6 = bce_loss(make_binary(d6),labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	#print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('JUTS', 'images' + os.sep)
tra_label_dir = os.path.join('JUTS', 'masks' + os.sep)

image_ext = '.jpg'
label_ext = '.jpg'

model_dir = os.path.join(
    os.getcwd(), 
    'saved_models', 
    model_name + os.sep
)

epoch_num = 100
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
#print(train_images[0], train_masks[0])
print("---")
print("val_images: ", len(val_images))
print("val_masks: ", len(val_masks))
#print(val_images[0], val_masks[0])
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

# ------- 4. define optimizer --------
print("---define optimizer...")

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
epoch_start = 0

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('best.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    avg_iou = checkpoint['avg_iou']
    print(f"Resuming from epoch {epoch_start} with val loss {avg_val_loss:.3f}")
    
# ------- 5. training process --------
print("---start training...")

max_iou = 0

for epoch in range(epoch_start, epoch_num):
    print('Epoch {}/{}'.format(epoch+1, epoch_num))
    # TRAINING
    net.train()

    for i, data in enumerate(tqdm(train_loader)):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

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
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

    print("\n[Epoch: %3d/%3d, batch: %5d/%5d, ite: %d train loss: %3f, tar: %3f " % (
    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    # RUN VALIDATION
    net.eval()
    total_iou = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
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
            # loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            # running_loss += loss.data.item()
            # total_val_loss += loss.data.item()
            # running_tar_loss += loss2.data.item()

            # CONVERT LABEL TO BINARY MASK (NP)
            actual = labels_v.squeeze()
            actual = actual.cpu().data.numpy()
            actual = np.where(actual > 0.6, 1, 0).astype('bool')

            # CONVERT PREDICTION TO BINARY MASK (NP)
            predict = d0[:,0,:,:]
            predict = normPRED(predict)
            predict = predict.squeeze()
            predict = predict.cpu().data.numpy()
            predict = np.where(predict > 0.6, 1, 0).astype('bool')

            iou = calculate_iou(actual, predict)
            total_iou += iou

            del d0, d1, d2, d3, d4, d5, d6, actual, predict, iou#, loss2, loss
    
    avg_iou = total_iou / val_num

    print(f"avg iou: {avg_iou:.3f}")
    
    # SAVE BEST MODEL
    if avg_iou > max_iou:
        max_iou = avg_iou
        #PATH = model_dir + model_name+"_best_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
        PATH=model_dir + model_name+"_best.pth" #_{avg_val_loss}
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_iou': avg_iou,
        }, PATH)
        print(f"saved best model: {avg_iou}!")
    
    # SAVE LAST MODEL 
    #PATH = model_dir + model_name+"_last_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
    PATH=model_dir + model_name+"_last.pth"  #_{avg_val_loss}
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_iou': avg_iou,
    }, PATH)
    print("saved last model")

    # if ite_num % save_frq == 0:
    #     torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
    #     #torch.save(optimizer.state_dict(),, "some_file_name"))
    #     running_loss = 0.0
    #     running_tar_loss = 0.0
    #     net.train()  # resume train
    #     ite_num4val = 0

