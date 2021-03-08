#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.utils.data
from data_loader import *
from IPython import display

import pickle
import glob
import os
import logging
import time
from datetime import datetime
from ast import literal_eval
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

import reed_muller_modules
from reed_muller_modules.logging_utils import *

from opt_einsum import contract   # This is for faster torch.einsum
from reed_muller_modules.reedmuller_codebook import *
from reed_muller_modules.hadamard import *
from reed_muller_modules.comm_utils import *
from reed_muller_modules.logging_utils import *
from reed_muller_modules.all_functions import *
import reed_muller_modules.reedmuller_codebook as reedmuller_codebook

import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from itertools import combinations


parser = argparse.ArgumentParser(description='(m,1) dumer')

parser.add_argument('--m', type=int, default=8, help='reed muller code parameter m')

parser.add_argument('--batch_size', type=int, default=50000, help='size of the batches')
parser.add_argument('--hidden_size', type=int, default=64, help='neural network size')

parser.add_argument('--full_iterations', type=int, default=40000, help='full iterations')
parser.add_argument('--enc_train_iters', type=int, default=50, help='encoder iterations')

parser.add_argument('--enc_train_snr', type=float, default=-4., help='snr at enc are trained')



parser.add_argument('--loss_type', type=str, default='BCE', choices=['MSE', 'BCE'], help='loss function')

parser.add_argument('--gpu', type=int, default=0, help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

device = torch.device("cuda:{0}".format(args.gpu))
kwargs = {'num_workers': 4, 'pin_memory': False}


results_save_path = './Results/RM({0},1)_softmap/NN_EncFull_Skip+Dec_Dumer/Enc_snr_{1}/Batch_{2}'\
    .format(args.m, args.enc_train_snr, args.batch_size)
os.makedirs(results_save_path, exist_ok=True)
os.makedirs(results_save_path+'/Models', exist_ok = True)

def repetition_code_matrices(device, m=8):
    
    M_dict = {}
    
    for i in range(1, m):
        M_dict[i] = torch.ones(1, 2**i).to(device)
    
    return M_dict

repetition_M_dict = repetition_code_matrices(device, args.m)

print("Matrices required for repition code are defined!")


Mul_this_Matrix_Ind_Zero = Variable(torch.load('./data/{0}/Mul_this_matrix_Ind_Zero.pt'.format(args.m))).to(device)
Mul_this_Matrix_Ind_One = Variable(torch.load('./data/{0}/Mul_this_matrix_Ind_One.pt'.format(args.m))).to(device)

## Loading the MAP indices for plus and minus one
PlusOneIdx = torch.load('./data/{0}/CodebookIndex_this_matrix_Zero_PlusOne.pt'.format(args.m)).long().to(device)
MinusOneIdx = torch.load('./data/{0}/CodebookIndex_this_matrix_One_MinusOne.pt'.format(args.m)).long().to(device)

print(PlusOneIdx, "\n", MinusOneIdx)

RM_Class = reedmuller_codebook.ReedMuller(1, args.m)
msg_length = RM_Class.message_length()

Generator_Matrix = numpy_to_torch(RM_Class.Generator_Matrix[:, ::-1].copy())
Generator_Matrix_cuda = Generator_Matrix.to(device)


## this is important because if the standard bits are (u_0, u_1, u_2,...,u_m) then the bits in our Plotkin structure are
## (u_0, u_m, u_m-1,...., u_1), i.e, u_1= u_0, v_1 = u_m,....,v_m = u_1.
tree_bits_order_from_standard = [0] + list(range(args.m, 0, -1))


print(Generator_Matrix_cuda, "\n", tree_bits_order_from_standard)
print("Mul_One:", Mul_this_Matrix_Ind_One, "\n", "Mul_Zero:", Mul_this_Matrix_Ind_Zero )

def llr_info_bits(hadamard_transform_llr, order_of_RM1):

    # Load Mul_this_Matrix_Ind_One/Zero before
    # order_of_RM1 = 7
    #hadam_transf_of_llr is of shape (batch*num_sparse, 128)

    LLR_Info_bits = torch.zeros(hadamard_transform_llr.shape[0], order_of_RM1 + 1).to(device)

    # Take care of tuple
    max_1, _ = hadamard_transform_llr.max(1)
    min_1, _ = hadamard_transform_llr.min(1)

    LLR_Info_bits[:, 0] = max_1 + min_1 

    # modify this tomorrow morning
    
    max_zero, _ = torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , Mul_this_Matrix_Ind_Zero), 2)
    max_one, _ =  torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , Mul_this_Matrix_Ind_One), 2)

    LLR_Info_bits[:, 1:] = max_zero - max_one

    return LLR_Info_bits


def modified_llr_codeword(LLR_Info_bits):

    # Generator matrix of shape (m+1, 2^m) for RM(m, 1) code is needed here. So load it before hand
    # LLR_Info_bits is of shape (batch*num_sparse, m + 1)

    required_LLR_info = contract('ij , jk ->ikj', LLR_Info_bits, Generator_Matrix_cuda) # (batch*num_sparse, 2^m, m+1)

    sign_matrix = (-1)**((required_LLR_info < 0).sum(2)).float() # (batch*num_sparse, 2^m+1)

    min_abs_LLR_info, _= torch.min(torch.where(required_LLR_info==0., torch.max(required_LLR_info.abs())+1, required_LLR_info.abs()), dim = 2)

    return sign_matrix * min_abs_LLR_info


def compute_llr_soft_decoding(llr, order_of_RM1):
#     hadamard_transform_llr = hadamard_transform_cuda(llr)  # shape (batch_size , 128)

    hadamard_transform_llr = hadamard_transform_torch(llr)  # shape (batch_size , 128)
#     return  modified_llr_codeword(llr_info_bits(hadamard_transform_llr, m))
    return  llr_info_bits(hadamard_transform_llr, order_of_RM1)


def decoder_soft_FHT(corrupted_codewords, snr, order_of_RM1):
    llr = llr_awgn_channel_bpsk(corrupted_codewords, snr)
    predicted_llr = compute_llr_soft_decoding(llr, order_of_RM1) # This order is in standard
#     return predicted_llr[:, tree_bits_order_from_standard]
    return predicted_llr



def rm_encoder(msg_bits):
    msg_bits = 0.5-0.5*msg_bits
    randomly_gen_codebook = reed_muller_batch_encoding(msg_bits, Generator_Matrix_cuda)
    
    return 1-2*randomly_gen_codebook
######
## Functions
######

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)


def log_sum_exp(LLR_vector):

    sum_vector = LLR_vector.sum(dim=1, keepdim=True)
    sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)

    return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 


def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return res


def errors_bler(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate





class g_identity(nn.Module):
    def __init__(self):
        super(g_vector, self).__init__()
        self.fc = nn.Linear(1, 1, bias=False)

    def forward(self, y):

        return y

class g_vector(nn.Module):
    def __init__(self):
        super(g_vector, self).__init__()
        self.fc = nn.Linear(16, 1, bias=True)

    def forward(self, y):

        return self.fc(y)



class g_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(g_Full, self).__init__()
        
        self.input_size  = input_size
        
        self.half_input_size = int(input_size/2)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True)
        
        # self.skip = nn.Linear(3*self.half_input_size, self.hidden_size, bias=False)

    def forward(self, y):
        x = F.selu(self.fc1(y))  

        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x) +  y[:, :self.half_input_size]*y[:, self.half_input_size:]
        return x
    
class f_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(f_Full, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, y):
        x = F.selu(self.fc1(y))
        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
def power_constraint(codewords, gnet_top, power_constraint_type, training_mode):
    
    
    if power_constraint_type in ['soft_power_block','soft_power_bit']:
        
        this_mean = codewords.mean(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.mean()
        this_std  = codewords.std(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.std()

        if training_mode == 'train':          # Training
            power_constrained_codewords = (codewords - this_mean)*1.0 / this_std
            
            gnet_top.update_normstats_for_test(this_mean, this_std)

        elif training_mode == 'test':         # For inference
            power_constrained_codewords = (codewords - gnet_top.mean_scalar)*1.0/gnet_top.std_scalar

#         else:                                 # When updating the stat parameters of g2net. Just don't do anything
#             power_constrained_codewords = _

        return power_constrained_codewords
        
    
    elif power_constraint_type == 'hard_power_block':
        
        return F.normalize(codewords, p=2, dim=1)*np.sqrt(2**args.m)


    else: # 'hard_power_bit'
        
        return codewords/codewords.abs()

# Plotkin stuff
def encoder_Plotkin(msg_bits):
    
    #msg_bits is of shape (batch, m+1)
    
    u_level0 = msg_bits[:, 0:1]
    v_level0 = msg_bits[:, 1:2]
    
    for i in range(2, args.m+1):
        
        u_level0 = torch.cat([ u_level0,  u_level0 * v_level0], dim=1)
        v_level0 = msg_bits[:, i:i+1].mm(repetition_M_dict[i-1])

    u_levelm = torch.cat([u_level0, u_level0 * v_level0], dim=1)
    
    return u_levelm
    
    

def encoder_full(msg_bits, gnet_dict, power_constraint_type='hard_power_block', training_mode='train'):    #g_avector, g_bvector, 
    
    u_level0 = msg_bits[:, 0:1]
    v_level0 = msg_bits[:, 1:2]
    
    for i in range(2, args.m+1):
        
        u_level0 = torch.cat([ u_level0, gnet_dict[i-1](torch.cat([u_level0, v_level0], dim=1)) ], dim=1)
        v_level0 = msg_bits[:, i:i+1].mm(repetition_M_dict[i-1])

    u_levelm = torch.cat([u_level0, gnet_dict[args.m](torch.cat([u_level0, v_level0], dim=1))], dim=1)
        
    return power_constraint(u_levelm, gnet_dict[args.m], power_constraint_type, training_mode)



def awgn_channel(codewords, snr):
    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords)
    corrupted_codewords = codewords+noise_sigma * standard_Gaussian
    return corrupted_codewords

def decoder_dumer(corrupted_codewords, snr):
    
    noise_sigma = snr_db2sigma(snr)
    
    llrs = (2/noise_sigma**2)*corrupted_codewords
    Lu = llrs
    
    decoded_bits = torch.zeros(corrupted_codewords.shape[0], args.m+1).to(device)
    
    for i in range(args.m-1, -1, -1):
    
        Lv = log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
        
        v_hat = torch.sign(Lv)
        
        decoded_bits[:, i+1] = v_hat.squeeze(1)
        
        Lu = Lu[:, :2**i] + v_hat * Lu[:, 2**i:]

    
    u_1_hat = torch.sign(Lu)
    decoded_bits[:, 0] = u_1_hat.squeeze(1)
    
    return decoded_bits


def decoder_dumer_soft(corrupted_codewords, snr):
    
    noise_sigma = snr_db2sigma(snr)
    
    llrs = (2/noise_sigma**2)*corrupted_codewords
    Lu = llrs
    
    decoded_bits = torch.zeros(corrupted_codewords.shape[0], args.m+1).to(device)
    
    for i in range(args.m-1, -1, -1):
    
        Lv = log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
        
        v_hat = torch.tanh(Lv/2)
        
        decoded_bits[:, i+1] = v_hat.squeeze(1)
        
        Lu = Lu[:, :2**i] + v_hat * Lu[:, 2**i:]

    
    u_1_hat = torch.tanh(Lu/2)
    decoded_bits[:, 0] = u_1_hat.squeeze(1)
    
    return decoded_bits


def decoder_nn_full(corrupted_codewords, fnet_dict):
    
    Lu = corrupted_codewords
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], args.m+1).to(device)
    
    for i in range(args.m-1, -1 , -1):
        
        Lv = fnet_dict[2*(args.m-i)-1](Lu)
        
        decoded_llrs[:, i+1] = Lv.squeeze(1)
        
        Lu = fnet_dict[2*(args.m-i)](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2), Lv.unsqueeze(1).repeat(1, 2**i, 1)], dim=2)).squeeze(2)
    
    decoded_llrs[:, 0] = Lu.squeeze(1)
    
    
    return decoded_llrs


def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.)

def first_principle_soft_MAP(corrupted_codewords, codebook_PlusOne, codebook_MinusOne):
    
    # modify this tomorrow morning
    
    max_PlusOne, _ = torch.max(contract('lk, ijk ->  lij', corrupted_codewords, codebook_PlusOne), 2)
    max_MinusOne, _ =  torch.max(contract('lk, ijk ->  lij', corrupted_codewords, codebook_MinusOne), 2)
    
    return max_PlusOne - max_MinusOne

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32).reshape(-1)
all_msg_bits = []

for i in range(2**(args.m+1)-1, -1, -1):
    all_msg_bits.append(bin_array(i,args.m+1)*2-1)
    
    
all_msg_bits = torch.tensor(np.array(all_msg_bits)).to(device)


# msg_bits = 2 * (torch.rand(args.full_iterations * args.batch_size, args.m+1) < 0.5).float() - 1
# Data_Generator = torch.utils.data.DataLoader(msg_bits, batch_size=args.batch_size , shuffle=True, **kwargs)

print("Data loading stuff is completed! \n")

gnet_dict = {}

for i in range(1, args.m+1):
    gnet_dict[i] =  g_Full(2*2**(i-1), args.hidden_size, 2**(i-1))

for i in range(1, args.m+1):
    gnet_dict[i].apply(weights_init)


# Now load them onto devices
for i in range(1, args.m+1):
    gnet_dict[i].to(device)


print("Models are loaded!")

enc_params = []
for i in range(1, args.m+1):
    enc_params += list(gnet_dict[i].parameters())


enc_optimizer = optim.Adam(enc_params, lr = 1e-5)#, momentum=0.9, nesterov=True) #, amsgrad=True)

criterion = nn.BCEWithLogitsLoss() if args.loss_type == 'BCE' else nn.MSELoss() # BinaryFocalLoss() #nn.BCEWithLogitsLoss() # nn.MSELoss() 

bers = []
losses = []

def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2): 
        distance = (row1-row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)

codebook_plotkin = encoder_Plotkin(all_msg_bits)
pairwise_dist_plotkin, d_min_plotkin = pairwise_distances(codebook_plotkin.data.cpu())
Gaussian_codebook = F.normalize(torch.randn(2**(args.m+1), 2**args.m), p=2, dim=1)*np.sqrt(2**args.m)
pairwise_dist_Gaussian, d_min_Gaussian = pairwise_distances(Gaussian_codebook)


import scipy.stats
rv = scipy.stats.chi(df=(2**(args.m)),scale=np.sqrt(2))


try:
    # for (k, msg_bits) in enumerate(Data_Generator):
    for k in range(args.full_iterations):
        
    
    
        start_time = time.time()
        msg_bits = 2 * (torch.rand(args.batch_size, args.m+1) < 0.5).float() - 1
        msg_bits = msg_bits.to(device)

        
            
                
        # Train Encoder
        for _ in range(args.enc_train_iters):
            
            # msg_bits = msg_bits.to(device)
            
            
            transmit_codewords = encoder_full(msg_bits, gnet_dict)      
            corrupted_codewords = awgn_channel(transmit_codewords, args.enc_train_snr)

            codebook_neural = encoder_full(all_msg_bits, gnet_dict)

            codebook_neural_PlusOne = codebook_neural[PlusOneIdx]
            codebook_neural_MinusOne = codebook_neural[MinusOneIdx]

            decoded_bits = (1/snr_db2sigma(args.enc_train_snr)**2)*first_principle_soft_MAP(corrupted_codewords, \
                                                               codebook_neural_PlusOne, codebook_neural_MinusOne)

            loss = criterion(decoded_bits, 0.5*msg_bits+0.5 )
            
            enc_optimizer.zero_grad()        
            loss.backward()
            enc_optimizer.step()
            
            ber = errors_ber(msg_bits, decoded_bits.sign()).item()
        
        
        
        bers.append(ber)

        losses.append(loss.item())
        
        

        if k % 10 == 0:
            print('[%d/%d] At %d dB, Loss: %.7f BER: %.7f' 
                % (k+1, args.full_iterations, args.enc_train_snr, loss.item(), ber))
            print("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))
            

        # Save the model for safety
        if (k+1) % 100 == 0:

            torch.save(dict(zip(['g{0}'.format(i) for i in range(1, args.m+1)], [gnet_dict[i].state_dict() for i in range(1, args.m+1)])),\
                    results_save_path+'/Models/Encoder_NN_{0}.pt'.format(k+1))
            
        
            plt.figure()
            plt.plot(bers)
            plt.plot(moving_average(bers, n=10))
            plt.savefig(results_save_path +'/training_ber.png')
            plt.close()

            plt.figure()
            plt.plot(losses)
            plt.plot(moving_average(losses, n=10))
            plt.savefig(results_save_path +'/training_losses.png')
            plt.close()

except KeyboardInterrupt:
    print('Graceful Exit')
else:
    print('Finished')







plt.figure()
plt.plot(bers)
plt.plot(moving_average(bers, n=10))
plt.savefig(results_save_path +'/training_ber.png')
plt.close()

plt.figure()
plt.plot(losses)
plt.plot(moving_average(losses, n=10))
plt.savefig(results_save_path +'/training_losses.png')
plt.close()

torch.save(dict(zip(['g{0}'.format(i) for i in range(1, args.m+1)], [gnet_dict[i].state_dict() for i in range(1, args.m+1)])),\
                    results_save_path+'/Models/Encoder_NN_{0}.pt'.format(k+1))
