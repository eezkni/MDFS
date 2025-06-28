import random
import torch
from PIL import Image
import glob

from model import EffNet, APL, prepare_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = EffNet().to(device)
model = APL().to(device)

flist = []
flist = glob.glob('./data/div500/*')+flist

random.shuffle(flist)
flist = flist[:1000]

feats_total = []
for i, f in enumerate(flist):
    x = prepare_image(Image.open(f).convert("RGB")).to(device)
    fname = f.split('/')[-1][:-4]
    feats_x = vgg(x)
    out, w = model(feats_x, select=1)
    feats_total.append(out)

feats_total = torch.cat(feats_total,dim=0)
torch.save(feats_total,'MDFS_weights.pth')
print('training finished!')
