import torch
import torch.nn as nn
from torchvision import models as tv
import torch.nn.functional as F
import math
from torchvision import transforms

class EffNet(torch.nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        model = tv.efficientnet_b7(pretrained=True).features#[:6]
        model.eval()
        # print(model)
        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        for param in self.parameters():
            param.requires_grad = False
        
        self.chns = [32, 48, 80, 160, 224]
        self.window_size = 3
        self.windows = self.create_window(self.window_size, self.window_size/3, 1)

    def gaussian(self,window_size, sigma, center = None):
        if center==None:
            center = window_size//2
        gauss = torch.Tensor([math.exp(-(x - center)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self,window_size, window_sigma, channel):
        _1D_window = self.gaussian(window_size, window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        # window = torch.ones_like(window)
        # window = window/window.sum(dim=[2,3],keepdim=True)
        return nn.Parameter(window,requires_grad=False)
      
    def get_features(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h        
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]#
        # for i in range(0,len(outs)):
        #     outs[i] = F.relu(outs[i])                
        return outs

    def construct_gau_pyramid(self, feats):
        f_pyr = [[] for i in range(len(feats))] 
        for i in range(len(feats)):
            f = feats[i]
            f_pyr[i].append(f)
            for j in range(i, len(feats)-1):
                win = self.windows.expand(f.shape[1], -1, -1, -1)
                pad = nn.ReflectionPad2d(win.shape[3]//2)
                f = F.conv2d(pad(f), win, stride=2, groups=win.shape[0])              
                if not f.shape[2] == feats[j+1].shape[2] or not f.shape[3] == feats[j+1].shape[3]:
                    f = F.interpolate(f, [feats[j+1].shape[2], feats[j+1].shape[3]], mode='bilinear', align_corners=True)
                f_pyr[j+1].append(f)

        for i in range(len(feats)):
            f_pyr[i] = torch.cat(f_pyr[i],dim=1)
        
        return f_pyr
       
    def forward(self, x):
        with torch.no_grad():
            feats_x = self.get_features(x)
            feats_last = feats_x[-1]
            feats_x = self.construct_gau_pyramid(feats_x[:-1])
        return torch.cat([feats_x[-1],feats_last],dim=1)

class APL(torch.nn.Module):
    def __init__(self):
        super(APL, self).__init__()

        self.chns = [32,48,80,160,224] # eff,196
                        
    def gaussian(self,window_size, sigma, center = None):
        if center==None:
            center = window_size//2
        gauss = torch.Tensor([math.exp(-(x - center)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self,window_size, window_sigma, channel):
        _1D_window = self.gaussian(window_size, window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

        return nn.Parameter(window,requires_grad=False)

    def forward(self, f, select=False):
        win_size = max(3, (min(f.shape[2],f.shape[3])//32)*2+1)
        stride = win_size//2

        windows = self.create_window(win_size, win_size/3, 1)
        win = windows.expand(f.shape[1], -1, -1, -1).to(f.device)
        pad = nn.ReflectionPad2d(0)
        f_mean = F.conv2d(pad(f), win, stride = stride, padding = 0, dilation=1, groups = f.shape[1])
        f_var = F.conv2d(pad(f**2), win, stride = stride, padding = 0, dilation=1, groups = f.shape[1]) - f_mean**2
        
        x_std = torch.mean(F.relu(f_var).sqrt(),dim=1,keepdim=True).squeeze()
        f_mean = f_mean.reshape(f_mean.shape[1],-1).permute(1,0)

        ps = 1 / (1 + torch.exp(-(x_std - x_std.mean()) / (x_std.std() + 1e-12)))  # [27, 41]

        if select:
            x_std = x_std.reshape(-1)
            ind = x_std>x_std.mean()
            f_mean = f_mean[ind,:]
                        
        f_mean = list(torch.split(f_mean,self.chns,dim=1))
        for i in range(len(f_mean)):
            f_mean[i] = f_mean[i]/(f_mean[i].norm(dim=1,keepdim=True)+1e-12)
        f_mean = torch.cat(f_mean,dim=1)

        return f_mean, ps

def prepare_image(image, resize = 0):
    if resize and min(image.size)>resize:
        image = transforms.functional.resize(image,resize)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


           