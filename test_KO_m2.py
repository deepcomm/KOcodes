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
from PIL import Image

import reed_muller_modules
from reed_muller_modules.logging_utils import *

from opt_einsum import contract   # This is for faster torch.einsum
from reed_muller_modules.reedmuller_codebook import *
from reed_muller_modules.hadamard import *
from reed_muller_modules.comm_utils import *
from reed_muller_modules.logging_utils import *
from reed_muller_modules.all_functions import *
# import reed_muller_modules.reedmuller_codebook as reedmuller_codebook

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

parser = argparse.ArgumentParser(description='(m,2) dumer')

parser.add_argument('--m', type=int, default=8, help='reed muller code parameter m')

parser.add_argument('--batch_size', type=int, default=100000, help='size of the batches')
parser.add_argument('--hidden_size', type=int, default=32, help='neural network size')


parser.add_argument('--enc_train_snr', type=float, default=0., help='snr at enc are trained')
parser.add_argument('--dec_train_snr', type=float, default=-2., help='snr at dec are trained')

parser.add_argument('--model_iters', type=int, default=201, help='model iterations')

parser.add_argument('--loss_type', type=str, default='BCE', choices=['MSE', 'BCE'], help='loss function')

parser.add_argument('--gpu', type=int, default=7, help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

device = torch.device("cuda:{0}".format(args.gpu))
kwargs = {'num_workers': 4, 'pin_memory': False}

results_save_path = './Results/RM({0},2)/fullNN_Enc+fullNN_Dec/Enc_snr_{1}_Dec_snr{2}/Batch_{3}'\
    .format(args.m, args.enc_train_snr,args.dec_train_snr, args.batch_size)
os.makedirs(results_save_path, exist_ok=True)
os.makedirs(results_save_path+'/Models', exist_ok = True)


def repetition_code_matrices(device, m=8):
    
    M_dict = {}
    
    for i in range(1, m):
        M_dict[i] = torch.ones(1, 2**i).to(device)
    
    return M_dict

def first_order_generator_matrices(device, m=8):
    
    G_dict = {}
    
    for i in range(1, m):
        
        RM_Class = ReedMuller(1, i if i>0 else 1)
        Generator_Matrix = numpy_to_torch(RM_Class.Generator_Matrix[:, ::-1].copy())
        G_dict[i] = Generator_Matrix.to(device)
    
    return G_dict


def first_order_Mul_matrices(device, m =8):
    
    Mul_Ind_Zero_dict, Mul_Ind_One_dict = {}, {}
    
    for i in range(1, m):
        # ## Loading the Mul_matrices
        Mul_Ind_Zero_dict[i] = Variable(torch.load('./data/{0}/Mul_this_matrix_Ind_Zero.pt'.format( i if i>0 else 1))).to(device)
        Mul_Ind_One_dict[i] = Variable(torch.load('./data/{0}/Mul_this_matrix_Ind_One.pt'.format( i if i>0 else 1))).to(device)
        
    
    return Mul_Ind_Zero_dict, Mul_Ind_One_dict

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32).reshape(-1)


def first_principle_soft_MAP(corrupted_codewords, codebook_PlusOne, codebook_MinusOne):
    
    # modify this tomorrow morning
    
    max_PlusOne, _ = torch.max(contract('lk, ijk ->  lij', corrupted_codewords, codebook_PlusOne), 2)
    max_MinusOne, _ =  torch.max(contract('lk, ijk ->  lij', corrupted_codewords, codebook_MinusOne), 2)
    
    return max_PlusOne - max_MinusOne

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

repetition_M_dict = repetition_code_matrices(device,args.m)

first_order_generator_dict = first_order_generator_matrices(device, args.m)

first_order_Mul_Ind_Zero_dict, first_order_Mul_Ind_One_dict = first_order_Mul_matrices(device, args.m)

print(first_order_generator_dict)

print(first_order_Mul_Ind_Zero_dict, "\n", first_order_Mul_Ind_One_dict)

print("Matrices required for first order code are defined!")

msg_lengths = [2]

for i in range(2, args.m+1):
    msg_lengths.append(i)

msg_bits_partition_indices = np.cumsum(msg_lengths)

code_dimension_k = msg_bits_partition_indices[-1]

print(msg_bits_partition_indices, code_dimension_k)

######
## Normal Functions
######

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)/np.sqrt(2)


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



####
## Neural Network Stuff
####




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
        x = self.fc4(x)+y[:, :, :self.half_input_size]*y[:,:, self.half_input_size:]
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


#####
## Order-1 Encoding & Decoding stuff
#####

def rm_encoder(msg_bits, Generator_Matrix):
    msg_bits = 0.5-0.5*msg_bits
    randomly_gen_codebook = reed_muller_batch_encoding(msg_bits, Generator_Matrix)
    
    return 1-2*randomly_gen_codebook






#####
##   Order-2
####
tree_bits_order_from_standard_dict = {}

for i in range(1, args.m):

    tree_bits_order_from_standard_dict[i] = [0] + list(range(i, 0, -1))
##### Order-1. Old stuff


# Leaves are repetition code
def first_order_encoder_Plotkin(msg_bits, order_of_RM1):
    
    
    u_level0 = msg_bits[:, 0:1]
    v_level0 = msg_bits[:, 1:2]
    
    for i in range(2, order_of_RM1+1):
        
        u_level0 = torch.cat([ u_level0,  u_level0 * v_level0], dim=1)
        v_level0 = msg_bits[:, i:i+1].mm(repetition_M_dict[i-1])

    u_levelm = torch.cat([u_level0, u_level0 * v_level0], dim=1)
    
    return u_levelm

def first_order_decoder_dumer(corrupted_codewords, snr):
    
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


def first_order_decoder_dumer_soft(corrupted_codewords, snr):
    
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


def first_order_decoder_nn_full(corrupted_codewords, fnet_dict):
    
    Lu = corrupted_codewords
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], args.m+1).to(device)
    
    for i in range(args.m-1, -1 , -1):
        
        Lv = fnet_dict[2*(args.m-i)-1](Lu)
        
        decoded_llrs[:, i+1] = Lv.squeeze(1)
        
        Lu = fnet_dict[2*(args.m-i)](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2), Lv.unsqueeze(1).repeat(1, 2**i, 1)], dim=2)).squeeze(2)
    

    return decoded_llrs






def llr_info_bits(hadamard_transform_llr, order_of_RM1):


    
    max_1, _ = hadamard_transform_llr.max(1, keepdim=True)
    min_1, _ = hadamard_transform_llr.min(1, keepdim=True)

    LLR_zero_column = max_1 + min_1 

    # modify this tomorrow morning
    
    max_zero, _ = torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , first_order_Mul_Ind_Zero_dict[order_of_RM1]), 2)
    max_one, _ =  torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , first_order_Mul_Ind_One_dict[order_of_RM1]), 2)

    LLR_remaining = max_zero - max_one

    return torch.cat([LLR_zero_column, LLR_remaining], dim=1)


def modified_llr_codeword(LLR_Info_bits, order_of_RM1):

    # Generator matrix of shape (m+1, 2^m) for RM(m, 1) code is needed here. So load it before hand
    # LLR_Info_bits is of shape (batch*num_sparse, m + 1)

    required_LLR_info = contract('ij , jk ->ikj', LLR_Info_bits, first_order_generator_dict[order_of_RM1]) # (batch*num_sparse, 2^m, m+1)

    sign_matrix = (-1)**((required_LLR_info < 0).sum(2)).float() # (batch*num_sparse, 2^m+1)

    min_abs_LLR_info, _= torch.min(torch.where(required_LLR_info==0., torch.max(required_LLR_info.abs())+1, required_LLR_info.abs()), dim = 2)

    return sign_matrix * min_abs_LLR_info


def FirstOrder_SoftFHT_InfoBits_decoder(llr, order_of_RM1):
    
    hadamard_transform_llr = hadamard_transform_torch(llr)  # shape (batch_size , 128)    
#     hadamard_transform_llr = hadamard_transform_cuda(llr)  # shape (batch_size , 128)    

    return llr_info_bits(hadamard_transform_llr, order_of_RM1) #     return  modified_llr_codeword(llr_info_bits(hadamard_transform_llr, m))


def FirstOrder_SoftFHT_LLR_decoder(llr, order_of_RM1, normalize=False):

    hadamard_transform_llr = hadamard_transform_torch(llr)  # shape (batch_size , 128)
#     hadamard_transform_llr = hadamard_transform_cuda(llr)  # shape (batch_size , 128)    

    return  modified_llr_codeword(llr_info_bits(hadamard_transform_llr, order_of_RM1), order_of_RM1)


def FirstOrder_SoftFHT_Codewords_decoder(corrupted_codewords, snr, order_of_RM1):
    
    llr = llr_awgn_channel_bpsk(corrupted_codewords, snr)
    
    predicted_llr = FirstOrder_SoftFHT_LLR_decoder(llr, order_of_RM1) # This order is in standard
    
    return predicted_llr







###########
## Order 2 --------------------------------------------------------------------------------------------------------------------
###########




def RM_22_Plotkin_encoder(msg_bits):

    u_level0 = torch.cat([msg_bits[:, 0:1], msg_bits[:, 0:1] * msg_bits[:, 1:2]], dim=1)
    v_level0 = torch.cat([msg_bits[:, 2:3], msg_bits[:, 2:3] * msg_bits[:, 3:4]], dim=1)

    return torch.cat([u_level0, u_level0 * v_level0], dim=1)


def correct_second_order_encoder_Plotkin_RM_leaves(msg_bits):
    
    #msg_bits is of shape (batch, m+1)
    

    ## This denotes the RM(2,2) right most node for u and the left RM(2,1) node for v.
    u_level0 = RM_22_Plotkin_encoder(msg_bits[:, :4])
    v_level0 = first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[1]: msg_bits_partition_indices[2]][:, tree_bits_order_from_standard_dict[2]], 2)
    
    for i in range(3, args.m):
        
        u_level0 = torch.cat([ u_level0,  u_level0 * v_level0], dim=1)
        v_level0 = \
        first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]][:, tree_bits_order_from_standard_dict[i]], i)

    u_levelm = torch.cat([u_level0, u_level0 * v_level0], dim=1)
    
    return u_levelm  

def correct_second_order_encoder_Neural_RM_leaves(msg_bits, gnet_dict, power_constraint_type='hard_power_block', training_mode='train'):    #g_avector, g_bvector, 
    ## This denotes the RM(2,2) right most node for u and the left RM(2,1) node for v.
    u_level0 = RM_22_Plotkin_encoder(msg_bits[:, :4])
    v_level0 = first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[1]: msg_bits_partition_indices[2]][:, tree_bits_order_from_standard_dict[2]], 2)
    
    for i in range(3, args.m):
        
        # u_level0 = torch.cat([ u_level0, gnet_dict[i](torch.cat([u_level0, v_level0], dim=1)) ], dim=1)s
        u_level0 = torch.cat([ u_level0, gnet_dict[i](torch.cat([u_level0.unsqueeze(2), v_level0.unsqueeze(2)], dim=2)).squeeze(2) ], dim=1)
        v_level0 = \
        first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]][:, tree_bits_order_from_standard_dict[i]], i)

    u_levelm = torch.cat([u_level0, gnet_dict[args.m](torch.cat([u_level0.unsqueeze(2), v_level0.unsqueeze(2)], dim=2)).squeeze(2)  ], dim=1)
        
    return power_constraint(u_levelm, None, power_constraint_type, training_mode)


def correct_second_order_OnlyTop_encoder_Neural_RM_leaves(msg_bits, gnet_dict, power_constraint_type='hard_power_block', training_mode='train'):    #g_avector, g_bvector, 
    
    ## This denotes the RM(2,2) right most node for u and the left RM(2,1) node for v.
    u_level0 = RM_22_Plotkin_encoder(msg_bits[:, :4])
    v_level0 = first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[1]: msg_bits_partition_indices[2]][:, tree_bits_order_from_standard_dict[2]], 2)
    
    for i in range(3, args.m):
        
        u_level0 = torch.cat([ u_level0,  u_level0 * v_level0], dim=1)
        v_level0 = \
        first_order_encoder_Plotkin(msg_bits[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]][:, tree_bits_order_from_standard_dict[i]], i)
    
    u_levelm = torch.cat([u_level0, gnet_dict[args.m](torch.cat([u_level0.unsqueeze(2), v_level0.unsqueeze(2)], dim=2)).squeeze(2)  ], dim=1)
        
    return power_constraint(u_levelm, None, power_constraint_type, training_mode)

##-----------------------------

## Loading the MAP indices for plus and minus one
PlusOneIdx_22_leaf = torch.load('./data/{0}/CodebookIndex_this_matrix_Zero_PlusOne.pt'.format(3)).long().to(device)
MinusOneIdx_22_leaf = torch.load('./data/{0}/CodebookIndex_this_matrix_One_MinusOne.pt'.format(3)).long().to(device)

# print(PlusOneIdx_22_leaf, "\n", MinusOneIdx_22_leaf)

all_msg_bits_22_leaf = []

for i in range(2**(4)-1, -1, -1):
    all_msg_bits_22_leaf.append(bin_array(i,4)*2-1)
    
    
all_msg_bits_22_leaf = torch.tensor(np.array(all_msg_bits_22_leaf)).to(device)

RM_22_codebook = RM_22_Plotkin_encoder(all_msg_bits_22_leaf)

RM_22_codebook_PlusOne = RM_22_codebook[PlusOneIdx_22_leaf]
RM_22_codebook_MinusOne = RM_22_codebook[MinusOneIdx_22_leaf]
    

def RM_22_SoftMAP_decoder(LLR):

    
    max_PlusOne, _ = torch.max(contract('lk, ijk ->  lij', LLR, RM_22_codebook_PlusOne), 2)
    max_MinusOne, _ =  torch.max(contract('lk, ijk ->  lij', LLR, RM_22_codebook_MinusOne), 2)
    
    return max_PlusOne - max_MinusOne


def correct_second_order_OnlyTop_decoder_nn_full(corrupted_codewords, fnet_dict, snr):

    noise_sigma = snr_db2sigma(snr)
    
    llrs = (2/noise_sigma**2)*corrupted_codewords
    Lu = llrs
    
    # Lu = corrupted_codewords
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], code_dimension_k).to(device)

    i = args.m-1

    Lv = fnet_dict[2*(args.m-i) - 1](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2)).squeeze(2)
    decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]] \
                            = FirstOrder_SoftFHT_InfoBits_decoder(Lv, i)
        
    v_hat = \
        torch.tanh(0.5*modified_llr_codeword(decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]], i))
        
    Lu =  fnet_dict[2*(args.m-i)](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2), Lv.unsqueeze(2), v_hat.unsqueeze(2)], dim=2)).squeeze(2)


    for i in range(args.m-2, 1 , -1):

        Lv = log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)) #.sum(dim=1, keepdim=True)
        
        
        decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]] = FirstOrder_SoftFHT_InfoBits_decoder(Lv, i)
        
        v_hat = \
            torch.tanh(0.5*modified_llr_codeword(decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]], i))
        
        Lu = Lu[:, :2**i] + v_hat * Lu[:, 2**i:]
        
    
    decoded_llrs[:, :4] = RM_22_SoftMAP_decoder(Lu)
    
    return decoded_llrs



def correct_second_order_decoder_nn_full(corrupted_codewords, fnet_dict):
    

    
    llrs = corrupted_codewords
    Lu = llrs
    
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], code_dimension_k).to(device)
    
    for i in range(args.m-1, 1 , -1):
        
        Lv = fnet_dict[2*(args.m-i) - 1](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2)).squeeze(2)+log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1))
        
        decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]]\
                                = FirstOrder_SoftFHT_InfoBits_decoder(Lv, i)
        
        v_hat = \
            torch.tanh(0.5*modified_llr_codeword(decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]], i))
        

        Lu =  fnet_dict[2*(args.m-i)](torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2), Lv.unsqueeze(2), v_hat.unsqueeze(2)], dim=2)).squeeze(2)+Lu[:, :2**i] + v_hat * Lu[:, 2**i:]
    
    
    decoded_llrs[:, :4] = RM_22_SoftMAP_decoder(Lu)
    
    return decoded_llrs


def correct_second_order_decoder_dumer_soft(corrupted_codewords, snr):
    
    noise_sigma = snr_db2sigma(snr)
    
    llrs = (2/noise_sigma**2)*corrupted_codewords
    Lu = llrs
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], code_dimension_k).to(device)
    
    for i in range(args.m-1, 1 , -1):
        
    
        Lv = log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)) 
        
        
        decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]] = FirstOrder_SoftFHT_InfoBits_decoder(Lv, i)
        
        v_hat = \
            torch.tanh(0.5*modified_llr_codeword(decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]], i))
        
        Lu = Lu[:, :2**i] + v_hat * Lu[:, 2**i:]
    
    decoded_llrs[:, :4] = RM_22_SoftMAP_decoder(Lu)
    
    return decoded_llrs




def correct_second_order_decoder_dumer(corrupted_codewords, snr):
    
    noise_sigma = snr_db2sigma(snr)
    
    llrs = (2/noise_sigma**2)*corrupted_codewords
    Lu = llrs
    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], code_dimension_k).to(device)
    
    for i in range(args.m-1, 1 , -1):
        
    
        Lv = log_sum_exp(torch.cat([Lu[:, :2**i].unsqueeze(2), Lu[:, 2**i:].unsqueeze(2)], dim=2).permute(0, 2, 1)) #.sum(dim=1, keepdim=True)
        
        
        decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]] = FirstOrder_SoftFHT_InfoBits_decoder(Lv, i).sign()
        
        v_hat = rm_encoder(decoded_llrs[:, msg_bits_partition_indices[i-1]: msg_bits_partition_indices[i]],\
                                    first_order_generator_dict[i])
        
        Lu = Lu[:, :2**i] + v_hat * Lu[:, 2**i:]
    
    decoded_llrs[:, :4] = RM_22_SoftMAP_decoder(Lu).sign() #FirstOrder_SoftFHT_InfoBits_decoder(Lu, 1).sign()
    
    return decoded_llrs

#------------------------------------------------------------------------------

# Leaves are Reed Muller codes


def awgn_channel(codewords, snr):
    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords)
    corrupted_codewords = codewords+noise_sigma * standard_Gaussian
    return corrupted_codewords


############################

def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n







# print(all_msg_bits)

gnet_dict = {}

for i in range(3, args.m+1):
    gnet_dict[i] =  g_Full(2, args.hidden_size, 1) #g_Full(2*2**(i-1), hidden_size, 2**(i-1))



    
fnet_dict = {}

for i in range(1, args.m-1):
    fnet_dict[2*i-1] = f_Full(2, args.hidden_size, 1) #f_Full(2**(m-i+1), hidden_size, 2**(m-i))
    fnet_dict[2*i] = f_Full(1+ 1+ 2, args.hidden_size, 1) #f_Full(1+ 1+ 2*2**(m-i), hidden_size, 1)
    




checkpoint1 = torch.load(results_save_path +'/Models/Encoder_NN_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)

for i in range(3, args.m+1):
    gnet_dict[i].load_state_dict(checkpoint1['g{0}'.format(i)])
    
checkpoint2 = torch.load(results_save_path +'/Models/Decoder_NN_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)

for i in range(1,args.m-1):
    fnet_dict[2*i-1].load_state_dict(checkpoint2['f{0}'.format(2*i-1)])
    fnet_dict[2*i].load_state_dict(checkpoint2['f{0}'.format(2*i)])

    
for i in range(3, args.m+1):
    gnet_dict[i].to(device)


for i in range(1, args.m-1):
    fnet_dict[2*i-1].to(device)
    fnet_dict[2*i].to(device)





print("Models are loaded!")

codebook_size = 1000
all_msg_bits = 2 * (torch.rand(codebook_size, code_dimension_k) < 0.5).float() - 1
    
    
all_msg_bits = all_msg_bits.to(device)


# read_file = np.load('codebooks_base.npz')
# all_msg_bits = torch.tensor(read_file['msg_bits']).to(device)


def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2): 
        distance = (row1-row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)


codebook_reusable_NN = correct_second_order_encoder_Neural_RM_leaves(all_msg_bits, gnet_dict) 
pairwise_dist_neural, d_min_reusable_NN = pairwise_distances(codebook_reusable_NN.data.cpu())
    

codebook_quantized_reuse_NN = codebook_reusable_NN.sign()
pairwise_dist_neural_quantized, d_min_quantized = pairwise_distances(codebook_quantized_reuse_NN.data.cpu())


codebook_plotkin = correct_second_order_encoder_Plotkin_RM_leaves(all_msg_bits)
pairwise_dist_plotkin, d_min_plotkin = pairwise_distances(codebook_plotkin.data.cpu())

print("Neural Codebook with d_min: {0: .4f} is \n {1}".format(d_min_reusable_NN, codebook_reusable_NN.data.cpu().numpy()))

print("Quantized Neural Codebook with d_min: {0: .4f} is \n {1}".format(d_min_quantized, codebook_quantized_reuse_NN.data.cpu().numpy()))

print("Plotkin Codebook with d_min: {0: .4f} is \n {1}".format(d_min_plotkin, codebook_plotkin))

Gaussian_codebook = F.normalize(torch.randn(codebook_size, 2**args.m), p=2, dim=1)*np.sqrt(2**args.m)
pairwise_dist_Gaussian, d_min_Gaussian = pairwise_distances(Gaussian_codebook)

total_pairwise_dist = len(pairwise_dist_neural)


min_stuff = np.min([np.min(pairwise_dist_neural), np.min(pairwise_dist_Gaussian)])
max_stuff = np.max([np.max(pairwise_dist_neural), np.max(pairwise_dist_Gaussian)])

bins = np.linspace(min_stuff, max_stuff, 1000)

n_neural, bins_neural = np.histogram(pairwise_dist_neural, bins=bins, density=True)#, density = False, bins=100, label='Neural Code: d_min={0:.2f}'.format(d_min_reusable_NN))

n_neural_quantized, bins_neural_quantized = np.histogram(pairwise_dist_neural_quantized, bins=bins, density=False)#, density = False, bins=100, label='Neural Code: d_min={0:.2f}'.format(d_min_reusable_NN))

n_RM, bins_RM  = np.histogram(pairwise_dist_plotkin, bins=bins, density=False)

n_Gaussian, bins_Gaussian = np.histogram(pairwise_dist_Gaussian, bins= bins,density=True)
n_RM  =  (1/total_pairwise_dist)*n_RM
n_neural_quantized = (1/total_pairwise_dist)*n_neural_quantized


from scipy.signal import savgol_filter
# n_Gaussian = savgol_filter(n_Gaussian, 101, 5)
# n_neural = savgol_filter(n_neural, 101, 5)

fig, ax = plt.subplots(figsize= (10, 7))

weight_polar = np.array([30, 80,96,104,112,120,128,136,144,152,160,176,192,208])
freq_polar = np.array([30., 1059,47250,166050,2126787,5976955,16916372,5978656,2126902,165437,47851,1045,37, 1])

plt.plot(np.sqrt(weight_polar)*2, freq_polar/(2**25), label='Polar', marker='o')
plt.xlim([16, 26])

plt.plot(bins_RM[:-1], n_RM, label='RM: Min dist = {0:.2f}'.format(d_min_plotkin), linewidth=2.0)
plt.plot(bins_neural[:-1], n_neural, label='Neural RM: Min dist = {0:.2f}'.format(d_min_reusable_NN), linewidth=2.0)
plt.plot(bins_neural_quantized[:-1], n_neural, label='Neural_quantized RM: Min dist = {0:.2f}'.format(d_min_quantized), linewidth=2.0)

plt.plot(bins_Gaussian[:-1], n_Gaussian, label='Random Gaussian: Min dist = {0:.2f}'.format(d_min_Gaussian), linewidth=2.0)
plt.xlabel("Pairwise distances", fontsize=16)
plt.ylabel("Probability density", fontsize=16)
plt.legend(loc='upper right', prop={'size': 15})
plt.title("Neural RM({0},2)".format(args.m), fontsize=16)
plt.savefig(results_save_path +'/Histogram11_{0}.pdf'.format(args.m))


np.savez(results_save_path+'/codebooks_awgn.npz', msg_bits = all_msg_bits.data.cpu().numpy(), nn_codebook=codebook_reusable_NN.data.cpu().numpy(), nn_quantized_codebook=codebook_quantized_reuse_NN.data.cpu().numpy(), plotkin_codebook=codebook_plotkin.data.cpu().numpy())







def test_all(msg_bits, snr):
    
    ## Common stuff
    
    noise_sigma = snr_db2sigma(snr)
    codewords_Plotkin = correct_second_order_encoder_Plotkin_RM_leaves(msg_bits)
    codewords_reuse_NN = correct_second_order_encoder_Neural_RM_leaves(msg_bits, gnet_dict)

    
    standard_Gaussian = torch.randn_like(codewords_reuse_NN)

    corrupted_codewords_Plotkin = codewords_Plotkin + noise_sigma * standard_Gaussian
    corrupted_codewords_reuse_NN = codewords_reuse_NN + noise_sigma * standard_Gaussian
    
    quantized_corrupted_codewords_reuse_NN = codewords_reuse_NN.sign() + noise_sigma * standard_Gaussian

    dumer_decoded_bits = correct_second_order_decoder_dumer(corrupted_codewords_Plotkin, snr)
    nn_decoded_bits = correct_second_order_decoder_nn_full(corrupted_codewords_reuse_NN, fnet_dict) 

    dumer_decoded_bits_for_nn = correct_second_order_decoder_nn_full(quantized_corrupted_codewords_reuse_NN, fnet_dict)
    nn_decoded_bits_for_plotkin = correct_second_order_decoder_nn_full(corrupted_codewords_Plotkin, fnet_dict) 

    
    ber_dumer = errors_ber(msg_bits, dumer_decoded_bits.sign()).item()
    ber_nn = errors_ber(msg_bits, nn_decoded_bits.sign()).item()

    bler_dumer = errors_bler(msg_bits, dumer_decoded_bits.sign()).item()
    bler_nn = errors_bler(msg_bits, nn_decoded_bits.sign()).item()

    ber_dumer_for_nn = errors_ber(msg_bits, dumer_decoded_bits_for_nn.sign()).item()
    ber_nn_for_plotkin = errors_ber(msg_bits, nn_decoded_bits_for_plotkin.sign()).item()


    
    ber_RPA_with_Reeds = 0
    bler_RPA_with_Reeds = 0

    return  ber_dumer, ber_nn, ber_dumer_for_nn, ber_nn_for_plotkin, bler_dumer, bler_nn, ber_RPA_with_Reeds, bler_RPA_with_Reeds


if args.m == 2:
    snr_range = np.linspace(0, 9, 9)-3
elif args.m == 3:
    snr_range = np.linspace(-3, 12, 16)-3
elif args.m == 4:
    snr_range = np.linspace(0, 10, 11)-3
elif args.m == 6:
    snr_range = np.linspace(0, 10, 11)-3
elif args.m == 7:
    snr_range = np.linspace(-7, 2, 10)-3
elif args.m == 8:
    snr_range = np.linspace(-7, 2, 10)-3
elif args.m == 9:
    snr_range = np.linspace(-10, 0, 11)-3






test_size = 1000000

bers_dumer_test = []
bers_nn_test = []
bers_dumer_for_nn_test = []
bers_nn_for_plotkin_test = []

blers_dumer_test = []
blers_nn_test = []

bers_RPA_with_Reeds = []
blers_RPA_with_Reeds = []

# os.makedirs(results)
results_file = os.path.join(results_save_path+'/ber_results_final.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')

#####
# Test Data
#####
Test_msg_bits = 2 * (torch.rand(test_size, code_dimension_k) < 0.5).float() - 1
Test_Data_Generator = torch.utils.data.DataLoader(Test_msg_bits, batch_size=1000 , shuffle=False, **kwargs)

num_test_batches = len(Test_Data_Generator)



for test_snr in tqdm(snr_range):

    bers_dumer, bers_nn = 0., 0.
    blers_dumer, blers_nn = 0., 0.

    bers_dumer_for_nn, bers_nn_for_plotkin= 0., 0.

    bers_rpa_reeds, blers_rpa_reeds = 0., 0.

    for (k, msg_bits) in enumerate(Test_Data_Generator):

        msg_bits = msg_bits.to(device)

        ber_dumer, ber_nn, ber_dumer_for_nn, ber_nn_for_plotkin, bler_dumer, bler_nn, ber_rpa_reeds, bler_rpa_reeds= test_all(msg_bits, snr=test_snr)

        bers_dumer += ber_dumer
        bers_nn += ber_nn
        bers_dumer_for_nn += ber_dumer_for_nn
        bers_nn_for_plotkin += ber_nn_for_plotkin
        blers_dumer += bler_dumer
        blers_nn += bler_nn

        bers_rpa_reeds += ber_rpa_reeds
        blers_rpa_reeds += bler_rpa_reeds



    bers_dumer /= num_test_batches
    bers_nn /= num_test_batches
    bers_dumer_for_nn /= num_test_batches
    bers_nn_for_plotkin /= num_test_batches
    blers_dumer /= num_test_batches
    blers_nn /= num_test_batches

    bers_rpa_reeds /= num_test_batches
    blers_rpa_reeds /= num_test_batches


    bers_dumer_test.append(bers_dumer)
    bers_nn_test.append(bers_nn)
    bers_dumer_for_nn_test.append(bers_dumer_for_nn)
    bers_nn_for_plotkin_test.append(bers_nn_for_plotkin)
    blers_dumer_test.append(blers_dumer)
    blers_nn_test.append(blers_nn)

    bers_RPA_with_Reeds.append(bers_rpa_reeds)
    blers_RPA_with_Reeds.append(blers_rpa_reeds)

    results.add(Test_SNR = test_snr, NN_BER = bers_nn, Plotkin_BER=bers_dumer, NN_BLER = blers_nn, Plotkin_BLER=blers_dumer, \
        RPA_BER=bers_rpa_reeds, RPA_BLER=blers_rpa_reeds)

    results.save()


print("BERs of Plotkin: \n {0}".format(bers_dumer_test))
print("BERs of NN Encoder: \n {0}".format(bers_nn_test))


print("BLERs of Plotkin: \n {0}".format(blers_dumer_test))

print("BLERs of NN Encoder: \n {0}".format(blers_nn_test))


from scipy.stats import norm

bers_optimal_concat = []

for test_snr in tqdm(snr_range):
    noise_sigma = snr_db2sigma(test_snr)
    bers_optimal_concat.append(norm.sf(np.sqrt(2**args.m/(code_dimension_k))/(noise_sigma), loc=0, scale=1))


plt.figure(figsize = (12,8))




plt.semilogy(snr_range, bers_dumer_test, label=" ",linewidth=2, marker='o', color='blue')
plt.semilogy(snr_range, bers_nn_test, label=" ", linewidth=2, marker='^', color='red')


plt.semilogy(snr_range, blers_dumer_test,linewidth=2, label=" ", marker='o' ,linestyle='dashed', color='blue')
plt.semilogy(snr_range, blers_nn_test,linewidth=2, marker='^', label=" ", linestyle='dashed', color='red' )



plt.grid()
# plt.xlabel("SNR (dB)", fontsize=16)
# plt.ylabel("BER", fontsize=16)
# plt.title("Trained at Enc_SNR = {0} dB and Dec_SNR = {1} dB".format(args.enc_train_snr, args.dec_train_snr))

plt.legend(prop={'size': 15})
# plt.show()
plt.savefig(results_save_path+ "/BER_at_test_SNRs_final.pdf")
plt.close()
