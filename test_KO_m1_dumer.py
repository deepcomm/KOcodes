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
# from IPython import display

import pickle
import glob
import os
import logging
import time
from datetime import datetime
from ast import literal_eval
import matplotlib
matplotlib.use('AGG')
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

parser.add_argument('--batch_size', type=int, default=20000, help='size of the batches')
parser.add_argument('--hidden_size', type=int, default=64, help='neural network size')

parser.add_argument('--full_iterations', type=int, default=10000, help='full iterations')
parser.add_argument('--enc_train_iters', type=int, default=50, help='encoder iterations')
parser.add_argument('--dec_train_iters', type=int, default=500, help='decoder iterations')

parser.add_argument('--enc_train_snr', type=float, default=-4., help='snr at enc are trained')
parser.add_argument('--dec_train_snr', type=float, default=-7., help='snr at dec are trained')


parser.add_argument('--power_constraint_type', type=str, default='hard_power_block', help='typer of power constraint')
parser.add_argument('--loss_type', type=str, default='BCE', choices=['MSE', 'BCE'], help='loss function')
parser.add_argument('--model_iters', type=int, default=0, help='model Iters')

parser.add_argument('--gpu', type=int, default=0, help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

device = torch.device("cuda:{0}".format(args.gpu))
kwargs = {'num_workers': 4, 'pin_memory': False}

def repetition_code_matrices(device, m=8):
    
    M_dict = {}
    
    for i in range(1, m):
        M_dict[i] = torch.ones(1, 2**i).to(device)
    
    return M_dict

repetition_M_dict = repetition_code_matrices(device, args.m)

print("Matrices required for repition code are defined!")

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
        x = self.fc4(x)+y[:, :self.half_input_size]*y[:, self.half_input_size:]
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
        
        Lv = fnet_dict[2*(args.m-i)-1](Lu)+log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
        
        v_hat = torch.tanh(Lv/2)

        decoded_llrs[:, i+1] = v_hat.squeeze(1)
        
        Lu = fnet_dict[2*(args.m-i)](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2), v_hat.unsqueeze(1).repeat(1, 2**i, 1)], dim=2)).squeeze(2)+Lu[:, :2**i] + v_hat * Lu[:, 2**i:]
    
    u_1_hat = torch.tanh(Lu/2)

    decoded_llrs[:, 0] = u_1_hat.squeeze(1)
    
    
    return decoded_llrs


def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# msg_bits = 2 * (torch.rand(args.full_iterations * args.batch_size, args.m+1) < 0.5).float() - 1
# Data_Generator = torch.utils.data.DataLoader(msg_bits, batch_size=args.batch_size , shuffle=True, **kwargs)

print("Data loading stuff is completed! \n")

gnet_dict = {}

for i in range(1, args.m+1):
    gnet_dict[i] =  g_Full(2*2**(i-1), args.hidden_size, 2**(i-1))


fnet_dict = {}

for i in range(1, args.m+1):
    fnet_dict[2*i-1] = f_Full(2**(args.m-i+1), args.hidden_size, 1)
    fnet_dict[2*i] = f_Full(1+ 1+ 1, args.hidden_size, 1)

    
#########
### Loading the models
#########
    
results_load_path = './Results/RM({0},1)/NN_EncFull_Skip+Dec_Dumer/Enc_snr_{1}_Dec_snr{2}/Batch_{3}'\
    .format(args.m, args.enc_train_snr,args.dec_train_snr, args.batch_size)

checkpoint1 = torch.load(results_load_path +'/Models/Encoder_NN_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)
print(checkpoint1)
for i in range(1, args.m+1):
    gnet_dict[i].load_state_dict(checkpoint1['g{0}'.format(i)])
    
checkpoint2 = torch.load(results_load_path +'/Models/Decoder_NN_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)

for i in range(1,args.m+1):
    fnet_dict[2*i-1].load_state_dict(checkpoint2['f{0}'.format(2*i-1)])
    fnet_dict[2*i].load_state_dict(checkpoint2['f{0}'.format(2*i)])

# Now load them onto devices
for i in range(1, args.m+1):
    gnet_dict[i].to(device)


for i in range(1, args.m+1):
    fnet_dict[2*i-1].to(device)
    fnet_dict[2*i].to(device)
print("Models are loaded!")



######
## Pairwise distances
######

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32).reshape(-1)

all_msg_bits = []

for i in range(2**(args.m+1)):
    all_msg_bits.append(bin_array(i,args.m+1)*2-1)
    
    
all_msg_bits = torch.tensor(np.array(all_msg_bits)).to(device)

print(all_msg_bits)


def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2): 
        distance = (row1-row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)


codebook_reusable_NN = encoder_full(all_msg_bits, gnet_dict, args.power_constraint_type, training_mode='test') # Just testing
pairwise_dist_neural, d_min_reusable_NN = pairwise_distances(codebook_reusable_NN.data.cpu())

# codebook_neural_PlusOne = codebook_reusable_NN[PlusOneIdx]
# codebook_neural_MinusOne = codebook_reusable_NN[MinusOneIdx]
    

codebook_quantized_reuse_NN = codebook_reusable_NN.sign()
_, d_min_quantized = pairwise_distances(codebook_quantized_reuse_NN.data.cpu())


codebook_plotkin = encoder_Plotkin(all_msg_bits)
pairwise_dist_plotkin, d_min_plotkin = pairwise_distances(codebook_plotkin.data.cpu())

print("Neural Codebook with d_min: {0: .4f} is \n {1}".format(d_min_reusable_NN, codebook_reusable_NN.data.cpu().numpy()))

print("Quantized Neural Codebook with d_min: {0: .4f} is \n {1}".format(d_min_quantized, codebook_quantized_reuse_NN.data.cpu().numpy()))

print("Plotkin Codebook with d_min: {0: .4f} is \n {1}".format(d_min_plotkin, codebook_plotkin))


Gaussian_codebook = F.normalize(torch.randn(2**(args.m+1), 2**args.m), p=2, dim=1)*np.sqrt(2**args.m)
pairwise_dist_Gaussian, d_min_Gaussian = pairwise_distances(Gaussian_codebook)
print(Gaussian_codebook[1:3,:].pow(2).sum(1))



###


all_msg_bits_large = all_msg_bits.t().unsqueeze(0).repeat(1000, 1, 1).to(device)


def encoder_codebook(msg_bits, codebook ):
    msg_bits_large = msg_bits.unsqueeze(2).repeat(1, 1, 2**(args.m+1)).to(device)
    diff = (msg_bits_large - all_msg_bits_large).pow(2).sum(dim=1)
    idx = diff.argmin(dim=1, keepdim=False)
    return codebook[idx,:]

##########
### Histogram stuff
#########

total_pairwise_dist = len(pairwise_dist_neural)

print(total_pairwise_dist)
# print(total_pairwise_dist)

# if m == 6:
#     range_histogram = (8,16)
# elif m ==8:
#     range_histogram = (10, 25)

min_stuff = np.min([np.min(pairwise_dist_neural), np.min(pairwise_dist_Gaussian)])
max_stuff = np.max([np.max(pairwise_dist_neural), np.max(pairwise_dist_Gaussian)])

bins = np.linspace(min_stuff, max_stuff, 1000)
# bins = np.arange(np.floor(np.min(pairwise_dist_neural)),np.ceil(np.max(pairwise_dist_neural)))

n_neural, bins_neural = np.histogram(pairwise_dist_neural, bins=bins, density=True)#, density = False, bins=100, label='Neural Code: d_min={0:.2f}'.format(d_min_reusable_NN))

# print("Neural", n_neural, "\n Bins:",bins_neural)
print(n_neural.sum())
# print(np.all(np.diff(bins_neural)==1))

n_RM, bins_RM  = np.histogram(pairwise_dist_plotkin, bins=bins, density=False)

n_Gaussian, bins_Gaussian = np.histogram(pairwise_dist_Gaussian, bins= bins,\
                                                 density=True)
# print("Gaussian", n_Gaussian, "\n Bins:",bins_Gaussian)
print(n_Gaussian.sum())
# n_neural =  n_neural / n_neural.sum()
n_RM  =  (1/total_pairwise_dist)*n_RM*510/511
# n_Gaussian = n_Gaussian / n_Gaussian.sum()


# print(n_RM)

from scipy.signal import savgol_filter
n_Gaussian = savgol_filter(n_Gaussian, 101, 5)
n_neural = savgol_filter(n_neural, 101, 5)
# n_RM = savgol_filter(n_RM, 101, 5)

fig, ax = plt.subplots(figsize= (10, 7))




# ax.annotate('Min dist. of RM={0:.2f}'.format(d_min_plotkin), xy=(d_min_plotkin, 0.))#, xytext=(25.,0.05), arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot(bins_RM[:-1], n_RM, label='RM: Min dist = {0:.2f}'.format(d_min_plotkin), linewidth=2.0)
plt.plot(bins_neural[:-1], n_neural, label='Neural RM: Min dist = {0:.2f}'.format(d_min_reusable_NN), linewidth=2.0)
plt.plot(bins_Gaussian[:-1], n_Gaussian, label='Random Gaussian: Min dist = {0:.2f}'.format(d_min_Gaussian), linewidth=2.0)
plt.xlabel("Pairwise distances", fontsize=16)
plt.ylabel("Probability density/mass", fontsize=16)
plt.legend(loc='upper right', prop={'size': 15})
plt.title("Histogram of pairwise distances", fontsize=16)
plt.savefig(results_load_path+'/Histogram_.pdf')





########
### Testing stuff
########

batch_inflated_neural_codebook = codebook_reusable_NN.t().unsqueeze(0).repeat(1000, 1, 1)

batch_inflated_Plotkin_codebook = codebook_plotkin.t().unsqueeze(0).repeat(1000, 1, 1)
batch_inflated_Gaussian_codebook = Gaussian_codebook.t().unsqueeze(0).repeat(1000, 1, 1).to(device)

print(batch_inflated_neural_codebook.shape, batch_inflated_Plotkin_codebook.shape, batch_inflated_Gaussian_codebook.shape)

def decoder_MAP(corrupted_codewords, batch_inflated_codebook):
    
    corrupted_codewords_inflated = corrupted_codewords.unsqueeze(2).repeat(1, 1, 2**(args.m+1)) #Both are of shape (batch, 256, 512)
    
    diff = (corrupted_codewords_inflated - batch_inflated_codebook).pow(2).sum(dim=1)
    
    idx = diff.argmin(dim=1, keepdim=False) #(batch)
    
    decoded_bits = all_msg_bits[idx, :]
    
    return decoded_bits

def test_MAP(msg_bits, snr):
    
#     codewords_old_NN = encoder_nn_old(msg_bits, g1net, g2net)
    
    codewords_reuse_NN = encoder_full(msg_bits, gnet_dict, args.power_constraint_type, training_mode='test')
    codewords_Plotkin = encoder_Plotkin(msg_bits)

    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords_reuse_NN)
    
    corrupted_codewords_reuse_NN = codewords_reuse_NN + noise_sigma * standard_Gaussian
    corrupted_codewords_Plotkin = codewords_Plotkin + noise_sigma * standard_Gaussian
    
    dumer_decoded_bits = decoder_MAP(corrupted_codewords_Plotkin, batch_inflated_Plotkin_codebook)    
    
    nn_decoded_bits = decoder_MAP(corrupted_codewords_reuse_NN, batch_inflated_neural_codebook)
    
    
    ber_dumer = errors_ber(msg_bits, dumer_decoded_bits).item()
    

    ber_nn = errors_ber(msg_bits, nn_decoded_bits).item()

    return ber_dumer, ber_nn


def test_all(msg_bits, snr):
    
#     codewords_old_NN = encoder_nn_old(msg_bits, g1net, g2net)
    
    codewords_reuse_NN = encoder_full(msg_bits, gnet_dict, args.power_constraint_type, training_mode='test')
    codewords_Plotkin = encoder_Plotkin(msg_bits)

    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords_reuse_NN)
    
    corrupted_codewords_reuse_NN = codewords_reuse_NN + noise_sigma * standard_Gaussian
    corrupted_codewords_Plotkin = codewords_Plotkin + noise_sigma * standard_Gaussian
    
    dumer_decoded_bits = decoder_soft_FHT(corrupted_codewords_Plotkin, snr, m)[:, tree_bits_order_from_standard].sign() #_dumer    
    nn_decoded_bits = decoder_soft_FHT(corrupted_codewords_reuse_NN, snr, m)[:, tree_bits_order_from_standard].sign()

    
    ber_dumer = errors_ber(msg_bits, dumer_decoded_bits).item()
    ber_nn = errors_ber(msg_bits, nn_decoded_bits).item()

    return ber_dumer, ber_nn


def test_MAP_and_all(msg_bits, snr):
    
    ## Common stuff
    
    noise_sigma = snr_db2sigma(snr)
    
    
    codewords_Plotkin = encoder_Plotkin(msg_bits)
    codewords_reuse_NN = encoder_full(msg_bits, gnet_dict, args.power_constraint_type, training_mode='test')
    
    codewords_Gaussian = encoder_codebook(msg_bits, Gaussian_codebook).to(device)
    
    standard_Gaussian = torch.randn_like(codewords_reuse_NN)

    corrupted_codewords_Plotkin = codewords_Plotkin + noise_sigma * standard_Gaussian
    corrupted_codewords_reuse_NN = codewords_reuse_NN + noise_sigma * standard_Gaussian
    corrupted_codewords_Gaussian = codewords_Gaussian+noise_sigma * standard_Gaussian
    
    
    
    ### MAP stuff
    dumer_decoded_bits = decoder_MAP(corrupted_codewords_Plotkin, batch_inflated_Plotkin_codebook)    
    nn_decoded_bits = decoder_MAP(corrupted_codewords_reuse_NN, batch_inflated_neural_codebook)

    Gaussian_decoded_bits = decoder_MAP(corrupted_codewords_Gaussian, batch_inflated_Gaussian_codebook)
    
    ber_dumer_map = errors_ber(msg_bits, dumer_decoded_bits).item()
    ber_nn_map = errors_ber(msg_bits, nn_decoded_bits).item()
    
    ber_Gaussian_map = errors_ber(msg_bits, Gaussian_decoded_bits).item()
    
    
    bler_msg_dumer_map = errors_bler(msg_bits, dumer_decoded_bits).item()
    bler_msg_nn_map = errors_bler(msg_bits, nn_decoded_bits).item()
    
    ### Existing decoding algorithms' stuff
    
#     dumer_decoded_bits = decoder_soft_FHT(corrupted_codewords_Plotkin, snr, m)[:, tree_bits_order_from_standard].sign() #_dumer    
#     nn_decoded_bits = first_principle_soft_MAP(corrupted_codewords_reuse_NN, \
#                                                            codebook_neural_PlusOne, codebook_neural_MinusOne).sign()
    
    
    dumer_decoded_bits = decoder_dumer(corrupted_codewords_Plotkin, snr)    
    
    nn_decoded_bits = decoder_nn_full(corrupted_codewords_reuse_NN, fnet_dict).sign() 

    
    ber_dumer = errors_ber(msg_bits, dumer_decoded_bits).item()
    ber_nn = errors_ber(msg_bits, nn_decoded_bits).item()
    
    bler_dumer = errors_bler(msg_bits, dumer_decoded_bits).item()
    bler_nn = errors_bler(msg_bits, nn_decoded_bits).item()
    
    
#     bler_msg_dumer_map = errors_bler(msg_bits, dumer_decoded_bits).item()
#     bler_msg_nn_map = errors_bler(msg_bits, nn_decoded_bits).item()
    bler_msg_gaussian_map = errors_bler(msg_bits, Gaussian_decoded_bits).item()

    return ber_dumer_map, ber_nn_map, ber_dumer, ber_nn, bler_dumer, bler_nn, ber_Gaussian_map, bler_msg_dumer_map, bler_msg_nn_map, bler_msg_gaussian_map
    

#### Final testing stuff

snr_range = np.linspace(-6, 12 , 19) if args.m<=4 else np.linspace(-10., 0., 11) #9, 16)# 6, 13)
test_size = 100000

bers_dumer_test = []
bers_nn_test = []

blers_dumer_test = []
blers_nn_test = []

bers_dumer_test_map = []
bers_nn_test_map = []

blers_msg_dumer_map_test = []
blers_msg_nn_map_test = []

bers_Gaussian_map_test = []
blers_msg_gaussian_map_test = []



bersu1_nn_test = []
bersv1_nn_test = []
bersv2_nn_test = []




os.makedirs(results_load_path, exist_ok=True)
results_file = os.path.join(results_load_path +'/ber_results.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')

#####
# Test Data
#####
Test_msg_bits = 2 * (torch.rand(test_size, args.m+1) < 0.5).float() - 1
Test_Data_Generator = torch.utils.data.DataLoader(Test_msg_bits, batch_size=1000 , shuffle=False, **kwargs)

num_test_batches = len(Test_Data_Generator)

for test_snr in tqdm(snr_range):
    
    bers_dumer, bers_nn = 0., 0.
    bers_dumer_map, bers_nn_map = 0., 0.
    
    blers_dumer, blers_nn = 0., 0.
    bers_Gaussian_map, blers_msg_dumer_map, blers_msg_nn_map, blers_msg_gaussian_map = 0.,0.,0.,0.
    
    
    for (k, msg_bits) in enumerate(Test_Data_Generator):
    
        msg_bits = msg_bits.to(device)

        ber_dumer_map, ber_nn_map, ber_dumer, ber_nn, bler_dumer, bler_nn, ber_Gaussian_map, bler_msg_dumer_map, bler_msg_nn_map, bler_msg_gaussian_map  = test_MAP_and_all(msg_bits, snr=test_snr)
        
        bers_dumer_map += ber_dumer_map
        bers_nn_map += ber_nn_map
        bers_dumer += ber_dumer
        bers_nn += ber_nn
        blers_dumer += bler_dumer
        blers_nn += bler_nn
        
        bers_Gaussian_map +=ber_Gaussian_map
        blers_msg_dumer_map +=bler_msg_dumer_map
        blers_msg_nn_map +=bler_msg_nn_map
        blers_msg_gaussian_map +=bler_msg_gaussian_map


    bers_dumer_map /= num_test_batches
    bers_nn_map /= num_test_batches    
    bers_dumer /= num_test_batches
    bers_nn /= num_test_batches
    
    blers_dumer /= num_test_batches
    blers_nn /= num_test_batches
    
    bers_Gaussian_map /= num_test_batches
    blers_msg_dumer_map /= num_test_batches
    blers_msg_nn_map /= num_test_batches
    blers_msg_gaussian_map /= num_test_batches
    
    bers_dumer_test.append(bers_dumer)
    bers_nn_test.append(bers_nn)
    
    bers_dumer_test_map.append(bers_dumer_map)
    bers_nn_test_map.append(bers_nn_map)
    
    
    blers_dumer_test.append(blers_dumer)
    blers_nn_test.append(blers_nn)
    
    
    blers_msg_dumer_map_test.append(blers_msg_dumer_map)
    blers_msg_nn_map_test.append(blers_msg_nn_map)
    
    bers_Gaussian_map_test.append(bers_Gaussian_map)
    blers_msg_gaussian_map_test.append(blers_msg_gaussian_map)
    
    results.add(Test_SNR = test_snr, NN_BER = bers_nn, Plotkin_BER = bers_dumer,  NN_BLER = blers_nn , Plotkin_BLER = blers_dumer , NN_BER_MAP = bers_nn_map, Plotkin_BER_MAP = bers_dumer_map, RandGauss_BER_MAP = bers_Gaussian_map, RandGauss_BLER_MAP = blers_msg_gaussian_map)

    results.save()

    
### Plotting stuff

## BER
plt.figure(figsize = (12,8))

ok = 1
plt.semilogy(snr_range[:-ok], bers_dumer_test[:-ok], label="RM + Dumer", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_nn_test[:-ok], label="Neural RM + Neural Dumer", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_Gaussian_map_test[:-ok], label="Random Gaussian + MAP", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_dumer_test_map[:-ok], label="RM + MAP (Inference)", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_nn_test_map[:-ok], label="Neural RM + MAP (Inference)", marker='^', linewidth=1.5)

plt.grid()
plt.xlabel("SNR (dB)", fontsize=16)
plt.ylabel("Bit Error Rate", fontsize=16)
# plt.title("Trained at Enc_SNR = {0} dB and Dec_SNR = {1} dB".format(enc_train_snr, dec_train_snr))
plt.title("BER plot of Neural RM({0},1) codes: Trained at Enc:{1}dB, Dec:{2}dB".format(args.m, args.enc_train_snr, args.dec_train_snr))
plt.legend(prop={'size': 15})
plt.savefig(results_load_path + "/{0}_BER_at_Test_SNRs.pdf".format(args.m))


### BLER
plt.figure(figsize = (12,8))

ok = 1
plt.semilogy(snr_range[:-ok], blers_dumer_test[:-ok], label="RM + Dumer", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_nn_test[:-ok], label="Neural RM + Neural Dumer", marker='^', linewidth=1.5)
# plt.semilogy(snr_range[:-ok], blers_Gaussian_map_test[:-ok], label="Random Gaussian + MAP", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_msg_dumer_map_test[:-ok], label="RM + MAP (Inference)", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_msg_nn_map_test[:-ok], label="Neural RM + MAP (Inference)", marker='^', linewidth=1.5)

plt.grid()
plt.xlabel("SNR (dB)", fontsize=16)
plt.ylabel("Message bits-Block Error Rate", fontsize=16)
# plt.title("Trained at Enc_SNR = {0} dB and Dec_SNR = {1} dB".format(enc_train_snr, dec_train_snr))
plt.title("BLER plot of Neural RM({0},1) codes: Trained at Enc:{1}dB, Dec:{2}dB".format(args.m, args.enc_train_snr, args.dec_train_snr))
plt.legend(prop={'size': 15})
plt.savefig(results_load_path + "/{0}_BLER_at_Test_SNRs.pdf".format(args.m))





