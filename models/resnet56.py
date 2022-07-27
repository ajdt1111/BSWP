import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .stripe import *

__all__ = ['ResNet56']
slope = 0.01


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = FilterStripe_res(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = BatchNorm(planes)
        self.conv2 = FilterStripe_res(planes, out_planes, kernel_size=3, stride=1)
        self.bn2 = BatchNorm(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                FilterStripe_res(in_planes, out_planes, kernel_size=1, stride=stride),
                BatchNorm(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet56(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(ResNet56, self).__init__()
        if cfg is None:
            cfg = [16] * 9 + [32] * 9 + [64] * 9
        self.in_planes = 16
        self.conv1 = FilterStripe_res(3, 16)
        self.bn1 = BatchNorm(16)
        self.layer1 = self._make_layer(16, cfg[:9], 9, stride=1)
        self.layer2 = self._make_layer(32, cfg[9:18], 9, stride=2)
        self.layer3 = self._make_layer(64, cfg[18:], 9, stride=2)
        self.fc = Linear(64, num_classes)

    def _make_layer(self, out_planes, cfg, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(BasicBlock(self.in_planes, cfg[i], out_planes, stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


    def update_skeleton(self, sr, dic_threshold):
        for key, m in self.named_modules():
            if isinstance(m, FilterStripe_res) and 'downsample' not in key:
                out_mask = m.update_skeleton(sr, dic_threshold[key])
            elif 'layer' in key and 'bn1' in key: #mask BN
                m.update_mask(out_mask)


    def prune(self,  dic_threshold):
        for key, m in self.named_modules():
            if key.startswith('conv'):
                m._break(dic_threshold[key])
            elif isinstance(m, BasicBlock):
                key1=key+'.conv1'
                key2=key+'.conv2'
                if (m.conv1.FilterSkeleton.data.abs()>dic_threshold[key1]).sum()==0 or (m.conv2.FilterSkeleton.data.abs()>dic_threshold[key2]).sum()==0:
                    if key.startswith('layer1'):
                        self.layer1.add_module(key.split('.')[1],m.downsample)
                    elif key.startswith('layer2'):
                        self.layer2.add_module(key.split('.')[1],m.downsample)
                    elif key.startswith('layer3'):
                        self.layer3.add_module(key.split('.')[1],m.downsample)
                else:
                    out_mask = m.conv1.prune_out(dic_threshold[key1])
                    m.bn1.prune(out_mask)
                    m.conv2.prune_in(out_mask)
                    m.conv1._break(dic_threshold[key1])
                    m.conv2._break(dic_threshold[key2])



class cal_shape(nn.Module):
        def __init__(self,dic_threshold):
            super(cal_shape,self).__init__()
            self.dic_threshold=dic_threshold

        def forward(self, model , dic_threshold):
            self.dic_threshold=dic_threshold
            loss_stripe_var = 0.
            loss_filter_var = 0.
            loss_mean = 0.
            for key, m in model.named_modules():
                if isinstance(m, FilterStripe_res) and 'downsample' not in key:
                    stripe_var,filter_var,mean = m.cal_skeleton(self.dic_threshold[key])
                    loss_stripe_var = loss_stripe_var + stripe_var
                    loss_filter_var = loss_filter_var + filter_var
                    loss_mean = loss_mean + mean
            return loss_stripe_var, loss_filter_var , loss_mean


class tool():
    def __init__(self,use_function):
        super(tool, self).__init__()
        self.use_function = use_function


    def get_piecewise(self, rate_temp, threshold):
        if rate_temp <= 0.2:
            value = threshold
        elif rate_temp <= 0.25:
            value = round(threshold * 1.025, 8)
        elif rate_temp <= 0.3:
            value = round(threshold * 1.05, 8)
        elif rate_temp <= 0.35:
            value = round(threshold * 1.075, 8)
        elif rate_temp <= 0.4:
            value = round(threshold * 1.1, 8)
        elif rate_temp <= 0.45:
            value = round(threshold * 1.125, 8)
        else:
            value = round(threshold * 1.2, 8)
        return value

    def creat_dic(self, model,threshold):
        dic_threshold= {}
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe_res) and 'downsample' not in key:
                temp = {key:threshold}
                dic_threshold.update(temp)
        return dic_threshold

    def change_rate(self,model,threshold,dic_threshold):
        dic_threshold_new = {}
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe_res) and 'downsample' not in key:
                rate_temp = m.cal_skeleton_show(dic_threshold[key])
                rate_temp = rate_temp/9
                if self.use_function == 'continuous': #slope = 0.008
                    value = threshold + slope  * rate_temp / (2 - rate_temp) #convex function
                    value = value.detach()
                    dic_temp = {key: value}
                    dic_threshold_new.update(dic_temp)
                if self.use_function == 'piecewise':
                    value = self.get_piecewise(rate_temp,threshold)
                    dic_temp = {key: value}
                    dic_threshold_new.update(dic_temp)
        return dic_threshold_new









