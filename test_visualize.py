from utils.model import *
from utils.FragmentDataset import FragmentDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils.visualize import plot_join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dim_size', type=int, default=32) #分辨率：32/64
parser.add_argument('--model_path', type=str, default="G32_2_2025-01-13-01-22-37.pth") #模型权重路径
parser.add_argument('--selected_class', type=int, default=3) #选取class。0表示所有class
args = parser.parse_args()


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset=FragmentDataset(vox_path='data',vox_type='test',dim_size=args.dim_size,transform=None,selected_class=args.selected_class)

test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)  

if args.dim_size==64:
    G=Generator64().to(device)
if args.dim_size==32:
    G=Generator32().to(device)

G.load_state_dict(torch.load(args.model_path))

# idx=torch.randint(0,len(test_dataset),(1,)).item()
frag, vox, label, img_path = next(iter(test_loader))
vox=vox>0.5
vox=vox.float()
rate=torch.sum(frag).item()/torch.sum(vox).item()
print("fragment proportion:",rate) #输出碎片占比大小
frag=frag.to(device)
vox=vox.to(device)
label=label.to(device)
print(img_path) #输出文件路径
new=G(frag,label)

new=new.cpu().detach().numpy()
vox=vox.cpu().detach().numpy()
frag=frag.cpu().detach().numpy()

new=new.reshape(args.dim_size,args.dim_size,args.dim_size)
vox=vox.reshape(args.dim_size,args.dim_size,args.dim_size)
frag=frag.reshape(args.dim_size,args.dim_size,args.dim_size)

new=new>0.8

# print(sum(frag.flatten()))
# print(sum(vox.flatten()))
# print(max(vox.flatten()))
plot_join(new,frag)
plot_join(vox,frag)


# python test_visualize.py --dim_size 32 --model_path G32_2_2025-01-13-01-22-37.pth --selecte_class 3
# python test_visualize.py --dim_size 64 --model_path G64_16_2025-01-13-17-33-07.pth --selecte_class 8
