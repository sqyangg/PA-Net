# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from panet import PANet, PAResNetBottleneck, PABottleneck



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, dataset, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        
      
        self.base = PANet(block=PAResNetBottleneck, 
                              layers=[2, 2, 2, 2], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=True,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
         
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.neck = neck
        self.neck_feat = neck_feat
            
        if dataset == 'UT_HAR_data':
        #################### for UT_HAR dataset#####
            self.reshape = nn.Sequential(
                nn.Conv2d(1,3,7,stride=(3,1)),
                nn.ReLU(),
                #nn.MaxPool2d(2),
                nn.Conv2d(3,3,kernel_size=(10,11),stride=1),
                nn.ReLU()
                )
            self.num_classes = 7
        else:
        #################### for UT_HAR dataset#####
                
                self.reshape = nn.Sequential(
                    nn.Conv2d(3,3,(15,23),stride=(3,9)),
                    nn.ReLU(),
                    nn.Conv2d(3,3,kernel_size=(3,23),stride=1),
                    nn.ReLU()
                    )
                self.num_classes = 6
        

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            
        
    def forward(self, x):
        ###########################for UT_HAR dataset#####
        
        x = self.reshape(x)
        
        #############for UT_HAR dataset#####

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        #if self.training:
         #   cls_score = self.classifier(feat)
          #  return cls_score, global_feat  # global feature for triplet loss
        #else:
         #   if self.neck_feat == 'after':
                # print("Test with feature after BN")
          #      return feat
           # else:
                # print("Test with feature before BN")
            #    return global_feat

        cls_score = self.classifier(feat)
        return cls_score, global_feat  # global feature for triplet loss
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
