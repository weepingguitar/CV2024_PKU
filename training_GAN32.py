import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import os
from utils.model import *
from utils.FragmentDataset import FragmentDataset
from rich.progress import Progress
from utils.visualize import *
import argparse
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def gDiffLoss(fake, real):
    temp=torch.where(real==0,fake,20*(1-fake))/21
    loss=torch.mean(torch.sum(temp,dim=(1,2,3)))
    return loss

def gradient_penalty(y_pred, voxel_blend):
    gradients = torch.autograd.grad(outputs=y_pred, inputs=voxel_blend, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2) * 10
    return penalty 

class GAN32_trainer:
    def __init__(self,args):
        self.args = args
        self.load_data()
        self.init_loss()
        self.G=Generator32().to(args.device)
        self.D=Discriminator32().to(args.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.g_lr, betas=(args.g_beta1,args.g_beta2),eps=args.g_eps,weight_decay=args.g_weight_decay)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.d_lr, betas=(args.d_beta1,args.d_beta2),eps=args.d_eps,weight_decay=args.d_weight_decay) 

    def load_data(self):
        train_dataset=FragmentDataset(self.args.train_data_path,"train",32)
        test_dataset=FragmentDataset(self.args.test_data_path,"test",32)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        print("Data Loaded")

    def init_loss(self):
        self.G_loss1=gDiffLoss
        # self.G_loss1=torch.nn.BCELoss()
        self.G_loss2=lambda y_pred:torch.mean(y_pred)
        # self.G_loss2=nn.BCELoss().to(self.args.device)
        self.D_loss1=lambda y_pred:torch.mean(y_pred)
        # self.D_loss1=nn.BCELoss().to(self.args.device)
        self.D_loss2=lambda y_pred:-torch.mean(y_pred)
        # self.D_loss2=nn.BCELoss().to(self.args.device)
        self.D_loss3=gradient_penalty
        self.D_loss_fake=[]
        self.D_loss_real=[]
        self.D_loss_grad=[]
        self.G_loss_diff=[]
        self.G_loss_pred=[]
        self.criterion = nn.BCELoss()

    def blend(self,vox_fake,vox_real):
        n=vox_fake.shape[0]
        alpha=torch.rand(n,1,1,1).to(self.args.device)
        return alpha*vox_fake+(1-alpha)*vox_real

    def add(self,vox_fake,vox_real):
        return torch.clamp(vox_fake+vox_real,0,1)
    def train_D(self,voxel_complete,voxel_frag,label):
        self.D_optimizer.zero_grad()
        self.G_optimizer.zero_grad()

        voxel_pred=self.G(voxel_frag,label)
        voxel_new=self.add(voxel_pred,voxel_frag)
        # voxel_new = self.G(voxel_frag,label) # fragment is added in the model

        # voxel_blend=self.blend(voxel_pred,voxel_complete).requires_grad_(True)
        D_complete=self.D(voxel_complete,label)
        # D_complete=D_complete[~torch.isnan(D_complete)]
        D_fake=self.D(voxel_new,label)
        # D_fake=D_fake[~torch.isnan(D_fake)]
        y_real= torch.tensor(np.random.uniform(0.9, 0.99, (voxel_complete.shape[0],1))).to(self.args.device).float()
        y_fake= torch.tensor(np.random.uniform(0.01, 0.1, (voxel_complete.shape[0],1))).to(self.args.device).float()

        # D_complete = torch.nan_to_num(D_complete, nan=0.0, posinf=1.0, neginf=0.0)
        # D_fake = torch.nan_to_num(D_fake, nan=0.0, posinf=1.0, neginf=0.0)

        loss_real=self.D_loss1(D_complete)
        # loss_real=self.D_loss1(D_complete,y_real)
        # loss_real=self.criterion(D_complete, y_real)
        loss_fake=self.D_loss2(D_fake)
        # loss_fake=self.D_loss2(D_fake,y_fake)
        # loss_fake=self.criterion(D_fake, y_fake)

        # loss_grad=self.D_loss3(self.D(voxel_blend,label),voxel_blend)

        self.D_loss_fake.append(loss_fake.item())
        self.D_loss_real.append(loss_real.item())
        # self.D_loss_grad.append(loss_grad.item())

        # loss=loss_real + loss_fake + torch.tensor(np.random.uniform(-1, 1)).to(self.args.device).float() # +loss_grad
        loss=loss_real + loss_fake # +loss_grad
        loss.backward()
        self.D_optimizer.step()
        # time.sleep(0.2)
    def train_G(self,voxel_complete,voxel_frag,label):
        self.D_optimizer.zero_grad()
        self.G_optimizer.zero_grad()

        voxel_pred=self.G(voxel_frag,label)
        voxel_new=self.add(voxel_pred,voxel_frag)
        # voxel_new = voxel_pred

        # if self.args.global_step%200==0:
        #     vox_pred_visual=voxel_pred[0].detach().cpu().numpy()
        #     vox_pred_visual=vox_pred_visual.round()
        #     vox_frag_visual=voxel_frag[0].detach().cpu().numpy()
        #     vox_complete_visual=voxel_complete[0].detach().cpu().numpy()

        #     plot_join(vox_pred_visual,vox_frag_visual)
        #     plot_join(vox_pred_visual,vox_complete_visual)
        #     print(label[0])
        # plot(vox_pred_visual, "vox_pred_visual")
        # plot(vox_complete_visual, "vox_complete_visual")

        loss_diff=self.G_loss1(voxel_new,voxel_complete)
        y_real= torch.tensor(np.random.uniform(0.9, 1.0, (voxel_complete.shape[0],1))).to(self.args.device).float()
        D_new=self.D(voxel_pred,label)
        loss_pred=self.G_loss2(D_new)
        # loss_pred=self.criterion(D_new, y_real)

        # loss_diff=loss_diff[~torch.isnan(loss_diff)]
        # loss_pred=loss_pred[~torch.isnan(loss_pred)]
        self.G_loss_diff.append(loss_diff.item())
        self.G_loss_pred.append(loss_pred.item())

        loss=loss_diff+loss_pred
        # if self.args.global_step>1500:
        #     loss=loss_diff*1e-4+loss_pred
        #     loss_pred.backward()
        # else:
        #     loss.backward()
        loss.backward()
        self.G_optimizer.step()
        time.sleep(0.2)

    def train_vae(self,voxel_complete,voxel_frag,label):
        self.G_optimizer.zero_grad()
        voxel_pred = self.G(voxel_frag, label)
        voxel_new = self.add(voxel_pred, voxel_frag)
        loss_diff = self.G_loss1(voxel_new, voxel_complete)
        self.G_loss_diff.append(loss_diff.item())
        loss_diff.backward()
        self.G_optimizer.step()
    def save(self,epoch):
        currenttime=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(self.G.state_dict(), f'G32_{epoch}_{currenttime}.pth')
        torch.save(self.D.state_dict(), f'D32_{epoch}_{currenttime}.pth')
        print("Model Saved")

    def save_vae(self,epoch):
        currenttime=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(self.G.state_dict(), f'G32VAE_{epoch}_{currenttime}.pth')
        print("Model Saved")

    def load_model(self,G_path,D_path):
        self.G.load_state_dict(torch.load(G_path))
        self.D.load_state_dict(torch.load(D_path))
        print("D and G Loaded")

    def load_g(self,G_path):
        self.G.load_state_dict(torch.load(G_path))
        print("G Loaded")

def vaetrain(trainer:GAN32_trainer):
    currenttime=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(f'runs/VAE_{currenttime}')
    with Progress() as progress:
        task1 = progress.add_task(
            f"[red]Epoch Training({0}/{trainer.args.vae_epochs})...",
            total=trainer.args.epochs,
        )

        for epoch in range(1, trainer.args.vae_epochs + 1):
            task2 = progress.add_task(
                f"[green]Epoch {1} Stepping({0}/{len(trainer.train_loader)-1})...",
                total=len(trainer.train_loader) - 1,
            )

            for step, (frags, voxes, labels, _) in enumerate(trainer.train_loader):
                trainer.args.global_step += 1
                frags = frags.to(trainer.args.device)
                voxes = voxes.to(trainer.args.device)
                labels = labels.to(trainer.args.device)  # fixed: device bug
                voxes = voxes > 0.5
                voxes = voxes.float()
                trainer.train_vae(voxes, frags, labels)

                G_loss_diff = (
                    float("inf")
                    if len(trainer.G_loss_diff) == 0
                    else trainer.G_loss_diff[-1]
                )

                progress.update(
                    task2,
                    advance=1,
                    completed=step,
                    description=f"[green]Epoch {epoch} Stepping({step}/{len(trainer.train_loader)-1}), ({G_loss_diff:.2f})...",
                )
                writer.add_scalar("Loss/G_loss_diff", G_loss_diff, trainer.args.global_step)

            G_loss_diff = np.mean(trainer.G_loss_diff)

            progress.update(
                task1,
                completed=epoch,
                description=f"[red]Epoch Training({epoch}/{trainer.args.epochs}), ({G_loss_diff:.2f})...",
            )

            date = datetime.now().strftime("%y%m%d%H%M%S")
            trainer.save_vae(epoch)
            # trainer.save_Model("VAE" + str(epoch) + "-" + date + ".pt")
            # trainer.draw_loss()
            # trainer.test(epoch, 10, True)
def train(trainer:GAN32_trainer):
    currenttime=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer=SummaryWriter(f'runs/GAN32_{currenttime}')
    with Progress() as progress:
        task1 = progress.add_task(f"[red]Epoch Training({0}/{trainer.args.epochs})...",total=trainer.args.epochs)

        for epoch in range(1,trainer.args.epochs+1):
            task2=progress.add_task(f"[green]Epoch {1} Stepping({0}/{len(trainer.train_loader)-1})...",total=len(trainer.train_loader)-1)
            for step,(frags,voxels,labels,_) in enumerate(trainer.train_loader):
                trainer.args.global_step+=1

                frags=frags.to(trainer.args.device)
                voxels=voxels.to(trainer.args.device)
                labels=labels.to(trainer.args.device)

                # if trainer.args.global_step%trainer.args.g_steps==0 or trainer.args.global_step>85:
                #     trainer.train_G(voxels,frags,labels)
                # if trainer.args.global_step%2==0 or trainer.args.global_step<85:
                #     trainer.train_D(voxels,frags,labels)

                if trainer.args.global_step%trainer.args.g_steps==0:
                    trainer.train_G(voxels,frags,labels)
                trainer.train_D(voxels, frags, labels)
                # if(step==45):
                #     trainer.load_g("G32VAE_5_2025-01-12-17-05-09.pth")
                # if(step>=45):
                #     trainer.train_G(voxels,frags,labels)
                D_loss_fake = float("inf") if len(trainer.D_loss_fake) == 0 else trainer.D_loss_fake[-1]
                D_loss_real = float("inf") if len(trainer.D_loss_real) == 0 else trainer.D_loss_real[-1]
                G_loss_pred = float("inf") if len(trainer.G_loss_pred) == 0 else trainer.G_loss_pred[-1]
                G_loss_diff = float("inf") if len(trainer.G_loss_diff) == 0 else trainer.G_loss_diff[-1]

                progress.update(
                    task2,
                    advance=1,
                    completed=step,
                    description=f"[green]Epoch {epoch} Stepping({step}/{len(trainer.train_loader)-1}), ({D_loss_fake+D_loss_real},{D_loss_fake:.3f},{D_loss_real:.3f},{G_loss_pred:.3f},{G_loss_diff:.2f})...",
                )
                # writer.add_scalar("Loss/D_loss_fake", D_loss_fake, trainer.args.global_step)
                # writer.add_scalar("Loss/D_loss_real", D_loss_real, trainer.args.global_step)
                writer.add_scalar("Loss/G_loss_pred", G_loss_pred, trainer.args.global_step)
                writer.add_scalar("Loss/G_loss_diff", G_loss_diff, trainer.args.global_step)
                # trainer.save(epoch)

            D_loss_fake = np.mean(trainer.D_loss_fake)
            D_loss_real = np.mean(trainer.D_loss_real)
            G_loss_pred = np.mean(trainer.G_loss_pred)
            G_loss_diff = np.mean(trainer.G_loss_diff)

            progress.update(
                task1,
                completed=epoch,
                description=f"[red]Epoch Training({epoch}/{trainer.args.epochs}), ({D_loss_fake:.2f},{D_loss_real:.2f},{G_loss_pred:.2f},{G_loss_diff:.2f})...",
            )
            trainer.save(epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="./data")
    parser.add_argument("--test_data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--g_beta1", type=float, default=0.9)
    parser.add_argument("--g_beta2", type=float, default=0.999)
    parser.add_argument("--g_eps", type=float, default=1e-8)
    parser.add_argument("--g_weight_decay", type=float, default=1e-5)
    parser.add_argument("--d_beta1", type=float, default=0.9)
    parser.add_argument("--d_beta2", type=float, default=0.999)
    parser.add_argument("--d_eps", type=float, default=1e-8)
    parser.add_argument("--d_weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--vae", type=str, default="false")
    parser.add_argument("--vae_epochs", type=int, default=5)
    args = parser.parse_args()

    trainer=GAN32_trainer(args)
    # trainer.load_model("G32_1_2025-01-10-23-32-17.pth", "D32_1_2025-01-10-23-32-17.pth")
    if args.vae=="true":
        vaetrain(trainer)
    else:
        #trainer.load_g("G32VAE_5_2025-01-12-17-05-09.pth") 
        train(trainer)

if __name__ == "__main__":
    main()
