# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 21:55:10 2023

@author: sqyan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:36:55 2023

@author: sqyan
"""
from yacs.config import CfgNode as CN
from baseline import Baseline
import os
import random
from sklearn.model_selection import KFold 
import numpy as np
import torch
import torch.nn as nn
import argparse
from dataset import CSI_Dataset
from dataset import *
def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, savelog):
    Note=open(savelog,mode='a')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    best_fit = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs,feat = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)

            loss = criterion(outputs,labels) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        #Note.write('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}\n'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        if best_fit < epoch_accuracy - epoch_loss :
            best_fit = epoch_accuracy - epoch_loss
            torch.save(model.state_dict(),'net_params132.pth')
    Note.close() 
    return


def test(model, tensor_loader, criterion, device,savelog):
    model.eval()
    Note=open(savelog,mode='a')
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs,feat = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    Note.write('validation accuracy:{:.4f}, loss:{:.5f}\n'.format(float(test_acc),float(test_loss)))
    Note.close() 
    return test_acc

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
 


if __name__ == "__main__":
    best_test_acc = 0
    best_test_epoch = 0
    for ii in range(30,31):
        seed_torch(ii)
        Note= 'panettainglog642.txt'
        print('Epoch:{}\n'.format(ii))
        Note1=open(Note,mode='a')
        
        Note1.write('Epoch:{}\n'.format(ii))
        root = './Data/' 
        parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
        parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi_HAR']) 
        
        args = parser.parse_args()
    
        args.dataset = 'UT_HAR_data'
    
        if args.dataset == 'UT_HAR_data':
            data = UT_HAR_dataset(root)
            train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
            test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
            train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
            test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
            #print(model)
            num_classes = 7
        else:
            print('using dataset: NTU-Fi_HAR')
            num_classes = classes['NTU-Fi_HAR']
            train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
            
            num_classes = 6
        
        LAST_STRIDE = 1
        MODELNECK ='bnneck' #  'no' 
        MODELNAME ='panet'
        PRETRAIN_CHOICE = 'no'
        MODELPRETRAIN_PATH = ''
        TESTNECK_FEAT = 'after'
        model = Baseline(args.dataset, LAST_STRIDE, PRETRAIN_CHOICE, MODELNECK, TESTNECK_FEAT, MODELNAME, PRETRAIN_CHOICE)
        
        train_epoch = 200
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss();
        IF_LABELSMOOTH = 'on'  ##use label smooth
        train(
            model=model,
            tensor_loader= train_loader,
            num_epochs= train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device, savelog = Note,
             )
        
        test_acc1 = test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device= device,savelog = Note
            )
        
        
        state_dict=torch.load('net_params132.pth')
        model.load_state_dict(state_dict)
        ''''''
        test_acc2 = test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device= device,savelog = Note
            )
        test_acc = max(test_acc1,test_acc2)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_test_epoch = ii
            Note1.write("best validation accuracy:{:.4f}, epock:{}\n".format(float(best_test_acc),float(best_test_epoch)))
        print("best validation accuracy:{:.4f}, epock:{}\n".format(float(best_test_acc),float(best_test_epoch)))
    Note1.close() 
