import torch
import torch.nn as nn
from utils.model import *
from utils.visualize import *
from utils.FragmentDataset import FragmentDataset
import argparse
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
def DSC(fake,real):
    eps=1e-6
    intersection = torch.sum(fake * real)
    union = torch.sum(fake) + torch.sum(real)
    return 2 * intersection / (union+eps)

def MSE(fake,real):
    return torch.mean((fake-real)**2)

def JAC(fake,real):
    eps=1e-6
    intersection = torch.sum(fake * real)
    union = torch.sum(fake) + torch.sum(real) - intersection
    return (union-intersection) / (union+eps)
def test(args):
    if args.model=='gan32':      
        G=Generator32()      
        G.load_state_dict(torch.load(args.G_model_path))
        test_dataset=FragmentDataset('data', 'test',32)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    if args.model=='gan64':
        G=Generator64()
        G.load_state_dict(torch.load(args.G_model_path))
        test_dataset=FragmentDataset('data', 'test',64)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    G.eval()

    G.to(device)
    test_DSC=0
    value1520=[[] for _ in range(11)]
    value2030=[[] for _ in range(11)]
    value30100=[[] for _ in range(11)]
    for i in range(3):
        for step, (frags, voxels, labels, path) in enumerate(
        tqdm(test_loader, desc="Processing")
        ):

            frags=frags.to(device)
            voxels=voxels.to(device)
            labels=labels.to(device)
            fake_voxels=G(frags,labels)
            voxels=voxels>0.5

            rate=torch.sum(frags).item()/torch.sum(voxels).item()
            # print(rate)

            if args.loss=='DSC':
                fake_voxels = fake_voxels > 0.8
                value=DSC(fake_voxels,voxels).item()
            if args.loss=='MSE':
                voxels=voxels.float()
                value=MSE(fake_voxels,voxels).item()
            if args.loss=='JAC':
                fake_voxels = fake_voxels > 0.8
                value=JAC(fake_voxels,voxels).item()

            if rate>0.15 and rate<0.2:
                value1520[int(labels)].append(value)
            if rate>=0.2 and rate<0.3:
                value2030[int(labels)].append(value)
            if rate>=0.3:
                value30100[int(labels)].append(value)
                
        # test_DSC+=DSC(fake_voxels,voxels)/len(test_loader)
    print(args.model)
    for i in range(11):
        print('label:',i)
        
        print(f'15-20 {args.loss}:',sum(value1520[i])/len(value1520[i]))
        print(f'20-30 {args.loss}:',sum(value2030[i])/len(value2030[i]))
        print(f'30-100 {args.loss}:',sum(value30100[i])/len(value30100[i]))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gan32')

    parser.add_argument('--G_model_path', type=str, default='G32_2_2025-01-13-01-22-37.pth')
    parser.add_argument('--loss', type=str, default='MSE')
    args=parser.parse_args()
    test(args)
