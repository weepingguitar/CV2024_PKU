# Try to implement proper metric for test function

# Try to implement some post-processing methds for visual evaluation, Display your generated results
import numpy as np
import torch
from utils.model import Generator, Discriminator
from visualize import *

def posprocessing(fake, mesh_frag):
    # fake is the generated M*M*(1 or 4) output, try to recover a voxel from it 
    # design by yourself or you can also choose to ignore this function
    return 


# You can implement the below two functions to load checkpoints and visualize .vox files. Option choice

# define available_device

def load_generator(path_checkpoint):
    ## for evaluation?
    G_encode_decode = Generator().to(available_device) #  hyperparams need to be implemented
    checkpoint = torch.load(path_checkpoint, map_location=available_device)
    G_encode_decode.load_state_dict(checkpoint)
    G_encode_decode = G_encode_decode.eval()

    return G_encode_decode


def generate(model, vox_frag):
    '''
    generate model, doesn't guaruantee 100% correct
    '''
    mesh_frag = torch.Tensor(vox_frag).unsqueeze(0).float().to(available_device)
    output_g_encode = model.forward_encode(mesh_frag)
    fake = model.forward_decode(output_g_encode)
    fake = fake + (mesh_frag.unsqueeze(1))
    fake = fake.detach().cpu().numpy()
    mesh_frag = mesh_frag.detach().cpu().numpy()
    return fake, mesh_frag