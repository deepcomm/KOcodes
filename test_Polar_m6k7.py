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

from opt_einsum import contract  # This is for faster torch.einsum

import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from itertools import combinations


parser = argparse.ArgumentParser(description='(m,k) Polar')

parser.add_argument('--m', type=int, default=6, help='number of layers in a polar code m')

parser.add_argument('--batch_size', type=int, default=20000, help='size of the batches')
parser.add_argument('--hidden_size', type=int, default=64, help='neural network size')

parser.add_argument('--full_iterations', type=int, default=20000, help='full iterations')
parser.add_argument('--enc_train_iters', type=int, default=50, help='encoder iterations')
parser.add_argument('--dec_train_iters', type=int, default=500, help='decoder iterations')

parser.add_argument('--enc_train_snr', type=float, default=-0.5., help='snr at enc are trained')
parser.add_argument('--dec_train_snr', type=float, default=-2.5., help='snr at dec are trained')

parser.add_argument('--power_constraint_type', type=str, default='hard_power_block', help='typer of power constraint')
parser.add_argument('--loss_type', type=str, default='BCE', choices=['MSE', 'BCE'], help='loss function')

parser.add_argument('--gpu', type=int, default=0, help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()


device = torch.device("cuda:{0}".format(args.gpu))
# device = torch.device("cpu")
kwargs = {'num_workers': 4, 'pin_memory': False}


def repetition_code_matrices(device, m=8):
    M_dict = {}

    for i in range(1, m):
        M_dict[i] = torch.ones(1, 2 ** i).to(device)

    return M_dict


repetition_M_dict = repetition_code_matrices(device, args.m)


print("Matrices required for repition code are defined!")


######
## Functions
######

def snr_db2sigma(train_snr):
    return 10 ** (-train_snr * 1.0 / 20)


def log_sum_exp(LLR_vector):
    sum_vector = LLR_vector.sum(dim=1, keepdim=True)
    sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)

    return torch.logsumexp(sum_concat, dim=1) - torch.logsumexp(LLR_vector, dim=1)


def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = sum(sum(myOtherTensor)) / (myOtherTensor.shape[0] * myOtherTensor.shape[1])
    return res


def errors_bler(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits - X_test)).view([X_test.shape[0], X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0, axis=1) > 0) * 1.0 / (X_test.shape[0])
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

        self.input_size = input_size

        self.half_input_size = int(input_size / 2)

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True)

        self.skip = nn.Linear(3 * self.half_input_size, self.hidden_size, bias=False)

    def forward(self, y):
        x = F.selu(self.fc1(y))
        x = F.selu(self.fc2(x)) + self.skip(
            torch.cat([y, y[:, :self.half_input_size] * y[:, self.half_input_size:]], dim=1))

        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return x


class f_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(f_Full, self).__init__()
        self.input_size = input_size
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
    if power_constraint_type in ['soft_power_block', 'soft_power_bit']:

        this_mean = codewords.mean(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.mean()
        this_std = codewords.std(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.std()

        if training_mode == 'train':  # Training
            power_constrained_codewords = (codewords - this_mean) * 1.0 / this_std

            gnet_top.update_normstats_for_test(this_mean, this_std)

        elif training_mode == 'test':  # For inference
            power_constrained_codewords = (codewords - gnet_top.mean_scalar) * 1.0 / gnet_top.std_scalar


        return power_constrained_codewords


    elif power_constraint_type == 'hard_power_block':

        return F.normalize(codewords, p=2, dim=1) * np.sqrt(2 ** args.m)


    else:  

        return codewords / codewords.abs()


def encoder_Polar_Plotkin(msg_bits):

    u_level1 = torch.cat([msg_bits[:, 6:7], msg_bits[:, 6:7] * msg_bits[:, 5:6]], dim=1)
    v_level1 = torch.cat([msg_bits[:, 4:5], msg_bits[:, 4:5] * msg_bits[:, 3:4]], dim=1)

    for i in range(2, args.m - 1):
        u_level1 = torch.cat([u_level1, u_level1 * v_level1], dim=1)
        v_level1 = msg_bits[:, 4-i:5-i].mm(repetition_M_dict[i])

    u_level5 = torch.cat([u_level1, u_level1 * v_level1], dim=1)

    u_level6 = torch.cat([u_level5, u_level5], dim=1)

    return u_level6


def encoder_Polar_full(msg_bits, gnet_dict, power_constraint_type='hard_power_block',
                       training_mode='train'):  # g_avector, g_bvector,

    u_level1 = torch.cat([msg_bits[:, 6:7], gnet_dict[1, 'right'](torch.cat([msg_bits[:, 6:7], msg_bits[:, 5:6]], dim=1)) ], dim=1)
    v_level1 = torch.cat([msg_bits[:, 4:5], gnet_dict[1, 'left'](torch.cat([msg_bits[:, 4:5], msg_bits[:, 3:4]], dim=1))], dim=1)

    for i in range(2, args.m - 1):
        u_level1 = torch.cat([u_level1, gnet_dict[i](torch.cat([u_level1, v_level1], dim=1)) ], dim=1)
        v_level1 = msg_bits[:, 4-i:5-i].mm(repetition_M_dict[i])

    u_level5 = torch.cat([u_level1, gnet_dict[args.m-1](torch.cat([u_level1, v_level1], dim=1)) ], dim=1)

    u_level6 = torch.cat([u_level5, u_level5], dim=1) 

    return power_constraint(u_level6, gnet_dict[args.m], power_constraint_type, training_mode)


def awgn_channel(codewords, snr):
    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords)
    corrupted_codewords = codewords + noise_sigma * standard_Gaussian
    return corrupted_codewords

def decoder_Polar_SC(corrupted_codewords, snr):
    noise_sigma = snr_db2sigma(snr)

    llrs = (2 / noise_sigma ** 2) * corrupted_codewords
    Lu = llrs
    Lu = Lu[:, 32:] + Lu[:, :32]

    decoded_bits = torch.zeros(corrupted_codewords.shape[0], args.m + 1).to(device)

    for i in range(args.m - 2, 1, -1):
        Lv = log_sum_exp(torch.cat([Lu[:, :2 ** i].unsqueeze(2), Lu[:, 2 ** i:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
        v_hat = torch.sign(Lv)
        decoded_bits[:, 4 - i] = v_hat.squeeze(1)
        Lu = Lu[:, :2 ** i] + v_hat * Lu[:, 2 ** i:]


    Lu2 = Lu
    Lv1 = log_sum_exp(torch.cat([Lu2[:, 0:2].unsqueeze(2), Lu2[:, 2:4].unsqueeze(2)], dim=2).permute(0, 2, 1))
    L_u3 = log_sum_exp(torch.cat([Lv1[:, 0:1].unsqueeze(2), Lv1[:, 1:2].unsqueeze(2)], dim=2).permute(0, 2, 1))
    u3_hat = torch.sign(L_u3)
    decoded_bits[:, 3] = u3_hat.squeeze(1)

    L_u4 = Lv1[:, 0:1] + u3_hat * Lv1[:, 1:2]
    u4_hat = torch.sign(L_u4)
    decoded_bits[:, 4] = u4_hat.squeeze(1)

    v1_hat = torch.cat([decoded_bits[:, 4:5], decoded_bits[:, 4:5] * decoded_bits[:, 3:4]], dim=1)
    Lu1 = Lu2[:, 0:2] + v1_hat * Lu2[:, 2:4]
    L_u5 = log_sum_exp(torch.cat([Lu1[:, 0:1].unsqueeze(2), Lu1[:, 1:2].unsqueeze(2)], dim=2).permute(0, 2, 1))
    u5_hat = torch.sign(L_u5)
    decoded_bits[:, 5] = u5_hat.squeeze(1)

    L_u6 = Lu1[:, 0:1] + u5_hat * Lu1[:, 1:2]
    u6_hat = torch.sign(L_u6)
    decoded_bits[:, 6] = u6_hat.squeeze(1)

    return decoded_bits


def decoder_Polar_SC_soft(corrupted_codewords, snr):
    noise_sigma = snr_db2sigma(snr)

    llrs = (2 / noise_sigma ** 2) * corrupted_codewords
    Lu = llrs
    Lu = Lu[:, 32:] + Lu[:, :32]

    decoded_bits = torch.zeros(corrupted_codewords.shape[0], args.m + 1).to(device)

    for i in range(args.m - 2, 1, -1):
        Lv = log_sum_exp(torch.cat([Lu[:, :2 ** i].unsqueeze(2), Lu[:, 2 ** i:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
        v_hat = torch.tanh(Lv/2)
        decoded_bits[:, 4 - i] = v_hat.squeeze(1)
        Lu = Lu[:, :2 ** i] + v_hat * Lu[:, 2 ** i:]


    Lu2 = Lu
    Lv1 = log_sum_exp(torch.cat([Lu2[:, 0:2].unsqueeze(2), Lu2[:, 2:4].unsqueeze(2)], dim=2).permute(0, 2, 1))
    L_u3 = log_sum_exp(torch.cat([Lv1[:, 0:1].unsqueeze(2), Lv1[:, 1:2].unsqueeze(2)], dim=2).permute(0, 2, 1))
    u3_hat = torch.tanh(L_u3/2)
    decoded_bits[:, 3] = u3_hat.squeeze(1)

    L_u4 = Lv1[:, 0:1] + u3_hat * Lv1[:, 1:2]
    u4_hat = torch.tanh(L_u4/2)
    decoded_bits[:, 4] = u4_hat.squeeze(1)

    v1_hat = torch.cat([decoded_bits[:, 4:5], decoded_bits[:, 4:5] * decoded_bits[:, 3:4]], dim=1)
    Lu1 = Lu2[:, 0:2] + v1_hat * Lu2[:, 2:4]
    L_u5 = log_sum_exp(torch.cat([Lu1[:, 0:1].unsqueeze(2), Lu1[:, 1:2].unsqueeze(2)], dim=2).permute(0, 2, 1))
    u5_hat = torch.tanh(L_u5/2)
    decoded_bits[:, 5] = u5_hat.squeeze(1)

    L_u6 = Lu1[:, 0:1] + u5_hat * Lu1[:, 1:2]
    u6_hat = torch.tanh(L_u6/2)
    decoded_bits[:, 6] = u6_hat.squeeze(1)

    return decoded_bits


def decoder_Polar_nn_full(corrupted_codewords, fnet_dict):

    Lu = corrupted_codewords
    Lu = Lu[:, 32:] + Lu[:, :32]

    
    decoded_llrs = torch.zeros(corrupted_codewords.shape[0], args.m + 1).to(device)

    for i in range(args.m - 2, 1, -1):
        Lv = fnet_dict[i+1, 'left'](Lu)
        decoded_llrs[:, 4 - i] = Lv.squeeze(1)
        v_hat = torch.tanh(Lv/2)
        Lu = fnet_dict[i+1, 'right'](torch.cat([Lu[:, :2 ** i].unsqueeze(2), Lu[:, 2 ** i:].unsqueeze(2), v_hat.unsqueeze(1).repeat(1, 2 ** i, 1)],dim=2)).squeeze(2)


    Lu2 = Lu
    Lv1 = fnet_dict[2, 'left'](Lu2)
    L_u3 = fnet_dict[1, 'left', 'left'](Lv1)
    decoded_llrs[:, 3] = L_u3.squeeze(1)
    u3_hat = torch.tanh(0.5 * L_u3)

    L_u4 = fnet_dict[1, 'left', 'right'](torch.cat([Lv1[:, 0:1].unsqueeze(2), Lv1[:, 1:2].unsqueeze(2), u3_hat.unsqueeze(1).repeat(1, 1, 1)],dim=2)).squeeze(2)
    decoded_llrs[:, 4] = L_u4.squeeze(1)
    u4_hat = torch.tanh(0.5 * L_u4)

    v1_hat = torch.cat([u4_hat, gnet_dict[1, 'left'](torch.cat([torch.sign(L_u4),  torch.sign(L_u3)], dim=1)) ], dim=1)
    Lu1 = fnet_dict[2, 'right'](torch.cat([Lu2[:, :2].unsqueeze(2), Lu2[:, 2:].unsqueeze(2), v1_hat.unsqueeze(2)],dim=2)).squeeze(2)
    L_u5 = fnet_dict[1, 'right', 'left'](Lu1)
    decoded_llrs[:, 5] = L_u5.squeeze(1)
    u5_hat = torch.tanh(0.5 * L_u5)

    L_u6 = fnet_dict[1, 'right', 'right'](torch.cat([Lu1[:, 0:1].unsqueeze(2), Lu1[:, 1:2].unsqueeze(2), u5_hat.unsqueeze(1).repeat(1, 1, 1)],dim=2)).squeeze(2)
    decoded_llrs[:, 6] = L_u6.squeeze(1)


    return decoded_llrs


def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


print("Data loading stuff is completed! \n")


gnet_dict = {}
gnet_dict[1, 'left'] = g_Full(2, args.hidden_size, 1)
gnet_dict[1, 'right'] = g_Full(2, args.hidden_size, 1)
for i in range(2, args.m + 1):
    gnet_dict[i] = g_Full(2 * 2 ** (i - 1), args.hidden_size, 2 ** (i - 1))

fnet_dict = {}
for i in range(3, 6):
    fnet_dict[i, 'left'] = f_Full(2 ** i, args.hidden_size, 1)
    fnet_dict[i, 'right'] = f_Full(1 + 1 + 1, args.hidden_size, 1)

fnet_dict[2, 'left'] = f_Full(4, args.hidden_size, 2)
fnet_dict[2, 'right'] = f_Full(1 + 1 + 1, args.hidden_size, 1)

fnet_dict[1, 'left', 'left'] = f_Full(2, args.hidden_size, 1)
fnet_dict[1, 'left', 'right'] = f_Full(1 + 1 + 1, args.hidden_size, 1)

fnet_dict[1, 'right', 'left'] = f_Full(2, args.hidden_size, 1)
fnet_dict[1, 'right', 'right'] = f_Full(1 + 1 + 1, args.hidden_size, 1)

#########
### Loading the models
#########

results_load_path = './Neural_Plotkin_Results/Polar({0},{1})_2ndModel/NN_EncFull_Skip+Dec_SC/Enc_snr_{2}_Dec_snr{3}/Batch_{4}' \
    .format(2**args.m, args.m+1, args.enc_train_snr, args.dec_train_snr, args.batch_size)

checkpoint1 = torch.load(results_load_path + '/Models/Encoder_NN_4300.pt', map_location=lambda storage, loc: storage)

gnet_dict[1, 'left'].load_state_dict(checkpoint1['g1_left'])
gnet_dict[1, 'right'].load_state_dict(checkpoint1['g1_right'])
for i in range(2, args.m + 1):
    gnet_dict[i].load_state_dict(checkpoint1['g{0}'.format(i)])


checkpoint2 = torch.load(results_load_path + '/Models/Decoder_NN_4300.pt', map_location=lambda storage, loc: storage)

for i in range(2, args.m):
    fnet_dict[i, 'left'].load_state_dict(checkpoint2['f{0}_left'.format(i)])
    fnet_dict[i, 'right'].load_state_dict(checkpoint2['f{0}_right'.format(i)])
fnet_dict[1, 'left', 'left'].load_state_dict(checkpoint2['f1_left_left'])
fnet_dict[1, 'left', 'right'].load_state_dict(checkpoint2['f1_left_right'])
fnet_dict[1, 'right', 'left'].load_state_dict(checkpoint2['f1_right_left'])
fnet_dict[1, 'right', 'right'].load_state_dict(checkpoint2['f1_right_right'])

gnet_dict[1, 'left'].to(device)
gnet_dict[1, 'right'].to(device)
for i in range(2, args.m + 1):
    gnet_dict[i].to(device)

for i in range(2, 6):
    fnet_dict[i, 'left'].to(device)
    fnet_dict[i, 'right'].to(device)
fnet_dict[1, 'left', 'left'].to(device)
fnet_dict[1, 'left', 'right'].to(device)
fnet_dict[1, 'right', 'left'].to(device)
fnet_dict[1, 'right', 'right'].to(device)
print("Models are loaded!")


######
## Pairwise distances
######

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32).reshape(-1)


all_msg_bits = []

for i in range(2 ** (args.m + 1)):
    all_msg_bits.append(bin_array(i, args.m + 1) * 2 - 1)

all_msg_bits = torch.tensor(np.array(all_msg_bits)).to(device)

print(all_msg_bits)


def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2):
        distance = (row1 - row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)


codebook_reusable_NN = encoder_Polar_full(all_msg_bits, gnet_dict, args.power_constraint_type,
                                          training_mode='test')  # Just testing
pairwise_dist_neural, d_min_reusable_NN = pairwise_distances(codebook_reusable_NN.data.cpu())




codebook_quantized_reuse_NN = codebook_reusable_NN.sign()

codebook_plotkin = encoder_Polar_Plotkin(all_msg_bits)



Gaussian_codebook = F.normalize(torch.randn(2 ** (args.m + 1), 2 ** args.m), p=2, dim=1) * np.sqrt(2 ** args.m)



all_msg_bits_large = all_msg_bits.t().unsqueeze(0).repeat(1000, 1, 1).to(device)


def encoder_codebook(msg_bits, codebook):
    msg_bits_large = msg_bits.unsqueeze(2).repeat(1, 1, 2 ** (args.m + 1)).to(device)
    diff = (msg_bits_large - all_msg_bits_large).pow(2).sum(dim=1)
    idx = diff.argmin(dim=1, keepdim=False)
    return codebook[idx, :]



########
### Testing stuff
########

batch_inflated_neural_codebook = codebook_reusable_NN.t().unsqueeze(0).repeat(1000, 1, 1)

batch_inflated_Plotkin_codebook = codebook_plotkin.t().unsqueeze(0).repeat(1000, 1, 1)
batch_inflated_Gaussian_codebook = Gaussian_codebook.t().unsqueeze(0).repeat(1000, 1, 1).to(device)

print(batch_inflated_neural_codebook.shape, batch_inflated_Plotkin_codebook.shape,
      batch_inflated_Gaussian_codebook.shape)


def decoder_MAP(corrupted_codewords, batch_inflated_codebook):
    corrupted_codewords_inflated = corrupted_codewords.unsqueeze(2).repeat(1, 1, 2 ** (
            args.m + 1))  
    diff = (corrupted_codewords_inflated - batch_inflated_codebook).pow(2).sum(dim=1)

    idx = diff.argmin(dim=1, keepdim=False)  # (batch)

    decoded_bits = all_msg_bits[idx, :]

    return decoded_bits


def test_MAP_and_all(msg_bits, snr):
    ## Common stuff

    noise_sigma = snr_db2sigma(snr)

    codewords_Plotkin = encoder_Polar_Plotkin(msg_bits)
    codewords_reuse_NN = encoder_Polar_full(msg_bits, gnet_dict, args.power_constraint_type, training_mode='test')

    codewords_Gaussian = encoder_codebook(msg_bits, Gaussian_codebook).to(device)

    standard_Gaussian = torch.randn_like(codewords_reuse_NN)

    corrupted_codewords_Plotkin = codewords_Plotkin + noise_sigma * standard_Gaussian
    corrupted_codewords_reuse_NN = codewords_reuse_NN + noise_sigma * standard_Gaussian
    corrupted_codewords_Gaussian = codewords_Gaussian + noise_sigma * standard_Gaussian

    ### MAP stuff
    Plotkin_decoded_bits = decoder_MAP(corrupted_codewords_Plotkin, batch_inflated_Plotkin_codebook)
    nn_decoded_bits = decoder_MAP(corrupted_codewords_reuse_NN, batch_inflated_neural_codebook)

    Gaussian_decoded_bits = decoder_MAP(corrupted_codewords_Gaussian, batch_inflated_Gaussian_codebook)

    ber_Plotkin_map = errors_ber(msg_bits, Plotkin_decoded_bits).item()
    ber_nn_map = errors_ber(msg_bits, nn_decoded_bits).item()

    ber_Gaussian_map = errors_ber(msg_bits, Gaussian_decoded_bits).item()

    bler_msg_Plotkin_map = errors_bler(msg_bits, Plotkin_decoded_bits).item()
    bler_msg_nn_map = errors_bler(msg_bits, nn_decoded_bits).item()

    SC_decoded_bits = decoder_Polar_SC(corrupted_codewords_Plotkin, snr)

    nn_decoded_bits = decoder_Polar_nn_full(corrupted_codewords_reuse_NN, fnet_dict).sign()

    ber_SC = errors_ber(msg_bits, SC_decoded_bits).item()
    ber_nn = errors_ber(msg_bits, nn_decoded_bits).item()

    bler_SC = errors_bler(msg_bits, SC_decoded_bits).item()
    bler_nn = errors_bler(msg_bits, nn_decoded_bits).item()


    bler_msg_gaussian_map = errors_bler(msg_bits, Gaussian_decoded_bits).item()

    return ber_Plotkin_map, ber_nn_map, ber_SC, ber_nn, bler_SC, bler_nn, ber_Gaussian_map, bler_msg_Plotkin_map, bler_msg_nn_map, bler_msg_gaussian_map


#### Final testing stuff

snr_range = np.linspace(-6, 12, 19) if args.m <= 4 else np.linspace(-10., 2, 13) 
test_size = 1000000

bers_SC_test = []
bers_nn_test = []

blers_SC_test = []
blers_nn_test = []

bers_Plotkin_test_map = []
bers_nn_test_map = []

blers_msg_Plotkin_map_test = []
blers_msg_nn_map_test = []

bers_Gaussian_map_test = []
blers_msg_gaussian_map_test = []

bersu1_nn_test = []
bersv1_nn_test = []
bersv2_nn_test = []

os.makedirs(results_load_path, exist_ok=True)
results_file = os.path.join(results_load_path + '/ber_results.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')

#####
# Test Data
#####
Test_msg_bits = 2 * (torch.rand(test_size, args.m + 1) < 0.5).float() - 1
Test_Data_Generator = torch.utils.data.DataLoader(Test_msg_bits, batch_size=1000, shuffle=False, **kwargs)

num_test_batches = len(Test_Data_Generator)

for test_snr in tqdm(snr_range):

    bers_SC, bers_nn = 0., 0.
    bers_Plotkin_map, bers_nn_map = 0., 0.

    blers_SC, blers_nn = 0., 0.
    bers_Gaussian_map, blers_msg_Plotkin_map, blers_msg_nn_map, blers_msg_gaussian_map = 0., 0., 0., 0.

    for (k, msg_bits) in enumerate(Test_Data_Generator):
        msg_bits = msg_bits.to(device)

        ber_Plotkin_map, ber_nn_map, ber_SC, ber_nn, bler_SC, bler_nn, ber_Gaussian_map, bler_msg_Plotkin_map, bler_msg_nn_map, bler_msg_gaussian_map = test_MAP_and_all(
            msg_bits, snr=test_snr)


        bers_Plotkin_map += ber_Plotkin_map
        bers_nn_map += ber_nn_map
        bers_SC += ber_SC
        bers_nn += ber_nn
        blers_SC += bler_SC
        blers_nn += bler_nn

        bers_Gaussian_map += ber_Gaussian_map
        blers_msg_Plotkin_map += bler_msg_Plotkin_map
        blers_msg_nn_map += bler_msg_nn_map
        blers_msg_gaussian_map += bler_msg_gaussian_map

    bers_Plotkin_map /= num_test_batches
    bers_nn_map /= num_test_batches
    bers_SC /= num_test_batches
    bers_nn /= num_test_batches

    blers_SC /= num_test_batches
    blers_nn /= num_test_batches

    bers_Gaussian_map /= num_test_batches
    blers_msg_Plotkin_map /= num_test_batches
    blers_msg_nn_map /= num_test_batches
    blers_msg_gaussian_map /= num_test_batches

    bers_SC_test.append(bers_SC)
    bers_nn_test.append(bers_nn)

    bers_Plotkin_test_map.append(bers_Plotkin_map)
    bers_nn_test_map.append(bers_nn_map)

    blers_SC_test.append(blers_SC)
    blers_nn_test.append(blers_nn)

    blers_msg_Plotkin_map_test.append(blers_msg_Plotkin_map)
    blers_msg_nn_map_test.append(blers_msg_nn_map)

    bers_Gaussian_map_test.append(bers_Gaussian_map)
    blers_msg_gaussian_map_test.append(blers_msg_gaussian_map)

    results.add(Test_SNR=test_snr, NN_BER=bers_nn, Plotkin_BER=bers_SC, NN_BLER=blers_nn, Plotkin_BLER=blers_SC,
                NN_BER_MAP=bers_nn_map, Plotkin_BER_MAP=bers_Plotkin_map, NN_BLER_MAP=blers_msg_nn_map, Plotkin_BLER_MAP=blers_msg_Plotkin_map,
                RandGauss_BER_MAP=bers_Gaussian_map, RandGauss_BLER_MAP=blers_msg_gaussian_map)

    results.save()

### Plotting stuff

## BER
plt.figure(figsize=(12, 8))

ok = 1
plt.semilogy(snr_range[:-ok], bers_SC_test[:-ok], label="Polar + SC", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_nn_test[:-ok], label="Neural Polar + Neural SC", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_Gaussian_map_test[:-ok], label="Random Gaussian + MAP", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_Plotkin_test_map[:-ok], label="Polar + MAP (Inference)", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], bers_nn_test_map[:-ok], label="Neural Polar + MAP (Inference)", marker='^', linewidth=1.5)
plt.ylim(2*(10**-6), 0.4)

plt.grid()
plt.xlabel("SNR (dB)", fontsize=16)
plt.ylabel("Bit Error Rate", fontsize=16)
plt.title("BER plot of Neural Polar({0},{1}) codes: Trained at Enc:{2}dB, Dec:{3}dB with Batch Size: {4}".format(2**args.m, args.m+1, args.enc_train_snr,
                                                                                                                 args.dec_train_snr, args.batch_size))
plt.legend(prop={'size': 15})
plt.savefig(results_load_path + "/{0}_BER_at_Test_SNRs.pdf".format(args.m))

### BLER
plt.figure(figsize=(12, 8))

ok = 1
plt.semilogy(snr_range[:-ok], blers_SC_test[:-ok], label="Polar + SC", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_nn_test[:-ok], label="Neural Polar + Neural SC", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_msg_gaussian_map_test[:-ok], label="Random Gaussian + MAP", marker='^', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_msg_Plotkin_map_test[:-ok], label="Polar + MAP (Inference)", marker='o', linewidth=1.5)
plt.semilogy(snr_range[:-ok], blers_msg_nn_map_test[:-ok], label="Neural Polar + MAP (Inference)", marker='^',
             linewidth=1.5)
plt.ylim(2*(10**-6), 0.4)

plt.grid()
plt.xlabel("SNR (dB)", fontsize=16)
plt.ylabel("Message bits-Block Error Rate", fontsize=16)
plt.title("BLER plot of Neural Polar({0},{1}) codes: Trained at Enc:{2}dB, Dec:{3}dB with Batch Size: {4}".format(2**args.m, args.m+1, args.enc_train_snr,
                                                                                                                  args.dec_train_snr, args.batch_size))
plt.legend(prop={'size': 15})
plt.savefig(results_load_path + "/{0}_BLER_at_Test_SNRs.pdf".format(args.m))





