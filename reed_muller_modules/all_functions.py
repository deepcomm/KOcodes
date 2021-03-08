from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.utils.data


from opt_einsum import contract   # This is for faster torch.einsum

import math
import numpy as np

import itertools
from itertools import combinations

import os


#########################
### These are imported from comm_utils
#########################

# note there are a few definitions of SNR. In our result, we stick to the following SNR setup.
def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)


def snr_sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)


##################################


def to_var(x, can_I_use_cuda, requires_grad=False):
    
    """Converts torch tensor to variable."""
    
    if can_I_use_cuda:
        x = x.cuda()
    
    return Variable(x, requires_grad=requires_grad)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def numpy_to_torch(tensor):

    return torch.from_numpy(tensor).float()


def Return_Index(M, S, b, m):

    # All of them are numpy arrays. Returns the indices a \in {0,1}^m such that a_S_bar = b
    S_bar = np.setdiff1d(np.arange(m), S)

    return np.flatnonzero(((M[:, S_bar] + b) % 2).sum(1) == 0 )


def integer_to_binary(integer, binary_base):

    ans = format(integer, '0{0}b'.format(binary_base))

    ans = numpy_to_torch(np.array([int(each_bit) for each_bit in ans]))

    return ans


def all_integers_to_binary(m):

    M = torch.zeros(2**m, m)

    for i in range(M.shape[0]):
        M[i, :] = integer_to_binary(i, m)

    return M


def all_binary_binary_dot_products(m, can_I_use_cuda):

    all_dot_products_matrix = to_var(all_integers_to_binary(m), can_I_use_cuda)

    return all_dot_products_matrix.mm(all_dot_products_matrix.t()) % 2


def binary_to_integer(binary_string):

    return int(binary_string, 2)


def fixed_vector_permutation_of_indices(m, integer_corresp_permutation):

    new_indices = [i^integer_corresp_permutation for i in range(2**m)]

    return torch.eye(2**m)[new_indices, :]


def all_vectors_permutation_of_indices(m):
    
    # First one is dummy. Hence it's really (2**m -1 , 2**m, 2**(m-1))
    all_permutations_tensor = torch.zeros(2**m, 2**m, 2**m)

    for integer_corresp_permutation in range(1, 2**m):
        
        all_permutations_tensor[integer_corresp_permutation, :, :] = fixed_vector_permutation_of_indices(m, integer_corresp_permutation)

    return all_permutations_tensor


def fixed_vector_coset_projection_matrix(m, integer_corresp_projection):

    M = torch.zeros(2**m, 2**(m-1))

    current_column = 0

    for integer_corresp_coset in range(2**m):
        if sum(M[integer_corresp_coset, :]) == 0:
            M[integer_corresp_coset, current_column] = 1

            remaining_element_in_coset = integer_corresp_coset^integer_corresp_projection

            M[remaining_element_in_coset, current_column] = 1

            current_column += 1

    return M


def all_vectors_coset_projection_tensor(m):

    # First one is dummy. Hence it's really (2**m -1 , 2**m, 2**(m-1))
    all_projections_tensor = torch.zeros(2**m, 2**m, 2**(m-1))

    for integer_corresp_projection in range(1, 2**m):

        all_projections_tensor[integer_corresp_projection, :, :] = fixed_vector_coset_projection_matrix(m, integer_corresp_projection)
        
    return all_projections_tensor

#############################
## s-dim stuff
#############################

def do_xor(list_elems, shift_elem): 

    #candidate_elem = 0 is the default.

    return [elem^shift_elem for elem in list_elems]


def find_xor(B, binary_string):

    assert(len(B) == len(binary_string))

    ans = 0

    for (i, basis) in enumerate(B):
        if binary_string[i] == 1:
            ans = ans^basis

    return ans


def get_basis_for_all_s_dim_subspaces(m, s, shift_elem):

    basis_all_s_dim_subspaces = [do_xor(list(i), shift_elem) for i in combinations(set([2**(i) for i in range(m)]),s)]

    return basis_all_s_dim_subspaces


def get_subspace_given_basis(B):

    # B is of size s.
    s = len(B)

    all_binary_strings_of_length_s = all_integers_to_binary(s).long()

    subspace = []

    for i in range(all_binary_strings_of_length_s.shape[0]):
        subspace.append(find_xor(B, all_binary_strings_of_length_s[i, :]))

    return subspace


def fixed_s_subspace_coset_projection_matrix(m, s, s_dim_subspace):

    # Finds the coset projection matrix for the given subspace.

    # print(s_dim_subspace)
    M = torch.zeros(2**m, 2**(m-s))

    current_column = 0

    for integer_corresp_coset in range(2**m):

        if sum(M[integer_corresp_coset, :]) == 0:
        
            all_elements_in_the_coset = [integer_corresp_coset^subspace_vector for subspace_vector in s_dim_subspace]

            M[all_elements_in_the_coset, current_column] = 1

            current_column += 1

    return M


def all_s_subspace_coset_projection_tensor(m, s, shift_elem=0):

    # s = r-1 for RM direct projection to order-1 codes.
    assert(s <= m)

    basis_all_s_dim_subspaces = get_basis_for_all_s_dim_subspaces(m, s, shift_elem)

    m_choose_s  = len(basis_all_s_dim_subspaces)

    # We only have (m s) projections. Input LLR is of shape 2**m. Output LLR is of shape 2**(m-s)
    all_cordinate_s_dim_projections_tensor = torch.zeros(m_choose_s, 2**m, 2**(m-s))

    for num_projections in range(m_choose_s):

        all_cordinate_s_dim_projections_tensor[num_projections, :, :] = fixed_s_subspace_coset_projection_matrix(m, s, get_subspace_given_basis(basis_all_s_dim_subspaces[num_projections]))

    return all_cordinate_s_dim_projections_tensor


def find_code_projection_s_dim_subspace_coset_indices(M, s):

    # Input (m choose s, 2**m, 2**(m-s)). Output (m choose s, 2**s, 2**(m-s))
    # M is of shape (70, 256, 16). We want a tensor of shape (70, 16, 16) that stores the indices of non-zero elements along each column

    two_power_s = 2**s
    idx = torch.zeros(M.shape[0], two_power_s, M.shape[2]).long()

    for i in range(idx.shape[0]):
        for j in range(idx.shape[2]):
            idx[i, :, j] = (M[i, :, j] == 1.).nonzero().reshape(two_power_s)

    return idx


def find_s_dim_coset_friends_message_passing_and_backprojection(Code_idx):

    # Code_idx is of shape (m choose s, 2**s, 2**(m-s)). So we want an output of shape (m choose s, 2**m, 2**s-1), where we store the other coset friends for each of the 256 main indices

    m_choose_s = Code_idx.shape[0]
    two_power_m = Code_idx.shape[1] * Code_idx.shape[2]
    two_power_s = Code_idx.shape[1]
    # s = int(math.log(two_power_s, 2))

    LLR_idx = torch.zeros(m_choose_s, two_power_m, two_power_s - 1).long()
    Coset_idx_for_bit_indices = torch.zeros(m_choose_s, two_power_m).long()

    for i in range(m_choose_s):
        mat = Code_idx[i,:, :]
        for j in range(two_power_m):
            that_column = (mat == j).nonzero().reshape(2)[1].item()
            all_coset_friends = mat[:, that_column]
            # print(all_coset_friends)
            LLR_idx[i, j, :] = all_coset_friends[all_coset_friends != j]
            Coset_idx_for_bit_indices[i, j] = that_column

    return LLR_idx, Coset_idx_for_bit_indices


def llr_coset_projection_s_dim(llr, Code_idx, even_comb, odd_comb): 

    # LLR is of shape (1, 256)
    # Code_idx is of shape (70, 16, 16)
    # output is of shape (1, 70, 16)
    # even_comb is the set of even-sized combinations of 2**s
    # odd_comb is the set of odd-sized combinations of 2**s

    Big_LLR = llr[:, Code_idx] # Of shape (1, 70, 16, 16)

    numerator = 1

    for each_comb in even_comb:

        numerator += Big_LLR[:, :, each_comb, :].sum(2).exp() # (1, 70, 16)
    
    # print(numerator[0].min(), numerator[0].max())

    denominator = 0

    for each_comb in odd_comb:

        denominator += Big_LLR[:, :, each_comb, :].sum(2).exp()
    
    # print(denominator[0].min(), denominator[0].max())


    return torch.log(numerator/denominator) # (1, 70, 16)


def llr_message_passing_aggregation_s_dim(llr, LLR_idx, odd_comb, even_comb):
    
    # llr is of shape (1, 256)
    # LLR_idx is of shape (70, 256, 15)
    # odd_comb is the set of odd-sized combinations of 2**s - 1
    # even_comb is the set of even-sized combinations of 2**s - 1

    Big_LLR = llr[:, LLR_idx] # shape (1, 70, 256, 15)

    numerator = 0
    
    for each_comb in odd_comb:

        numerator += Big_LLR[:, :, :, each_comb].sum(3).exp()

    denominator = 1

    for each_comb in even_comb:

        denominator += Big_LLR[:, :, :, each_comb].sum(3).exp()

    return torch.log(numerator/denominator) # (1, 70, 256)


def log_sum_avoid_NaN(x, y):

    a = torch.max(x, y)
    b = torch.min(x, y)

    log_sum_standard = torch.log(1 + (x+y).exp()) - x - torch.log(1 + (y-x).exp() )

    # print("Original one:", log_sum_standard)
    
    ## Check for NaN or infty or -infty once here. 
    if (torch.isnan(log_sum_standard).sum() > 0) | ((log_sum_standard == float('-inf')).sum() > 0 )| ( (log_sum_standard == float('inf')).sum() > 0) :
        
        # print("Had to avoid NaNs!")
        # 80 for float32 and 707 for float64.
        big_threshold = 80. if log_sum_standard.dtype == torch.float32 else 700.

        idx_1 = (x + y > big_threshold)
        subset_1 = idx_1 & ((x-y).abs() < big_threshold)

        idx_2 = (x + y < -big_threshold)
        subset_2 = idx_2 & ((x-y).abs() < big_threshold)
        
        idx_3 = ((x - y).abs() > big_threshold) & ( (x+y).abs() < big_threshold )

        # Can be fastened
        if idx_1.sum() > 0 :

            if subset_1.sum() > 0:
                log_sum_standard[subset_1] = y[subset_1]- torch.log(1 + (y[subset_1] - x[subset_1]).exp() )
                # print("After 11 modification", log_sum_standard)
            
            if (idx_1 - subset_1).sum() > 0:
                log_sum_standard[idx_1 - subset_1] = b[idx_1 - subset_1]
                # print("After 12 modification", log_sum_standard)

        if idx_2.sum() > 0:

            if subset_2.sum() > 0:
                log_sum_standard[subset_2] = -x[subset_2]- torch.log(1 + (y[subset_2] - x[subset_2]).exp() )
                # print("After 21 modification", log_sum_standard)

            if (idx_2 - subset_2).sum() > 0:
                log_sum_standard[idx_2 - subset_2] = -a[idx_2 - subset_2]
                # print("After 22 modification", log_sum_standard)

        if idx_3.sum() > 0:
            
            log_sum_standard[idx_3] = torch.log(1 + (x[idx_3]+ y[idx_3]).exp() ) - a[idx_3]
            # print("After 3 modification", log_sum_standard)

    return log_sum_standard


def log_sum_avoid_zero_NaN(x, y):

    avoided_NaN = log_sum_avoid_NaN(x,y)

    zero_idx = (avoided_NaN == 0.)

    data_type = x.dtype

    if zero_idx.sum() > 0:

        # print("Had to avoid zeros!")

        x_subzero = x[zero_idx]
        y_subzero = y[zero_idx]

        nume = torch.relu(x_subzero + y_subzero)
        denom = torch.max(x_subzero , y_subzero)
        delta = 1e-7 if data_type == torch.float32 else 1e-16

        term_1 = 0.5 *( (-nume).exp() + (x_subzero + y_subzero - nume).exp() )
        term_2 = 0.5 * ( (x_subzero - denom).exp() + (y_subzero - denom).exp() )

        close_1 = torch.tensor( (term_1 - 1).abs() < delta, dtype= data_type).cuda()
        T_1 =  (term_1 - 1.) * close_1 + torch.log(term_1) * (1-close_1)
        
        close_2 = torch.tensor( (term_2 - 1).abs() < delta, dtype= data_type).cuda()
        T_2 =  (term_2 - 1.) * close_2 + torch.log(term_2) * (1-close_2)

        corrected_ans = nume - denom + T_1 - T_2

        further_zero = (corrected_ans == 0.)

        if further_zero.sum() > 0:

                x_sub_subzero = x_subzero[further_zero]
                y_sub_subzero = y_subzero[further_zero]

                positive_idx = ( x_sub_subzero + y_sub_subzero > 0.)

                spoiled_brat = torch.min(- x_sub_subzero, - y_sub_subzero)

                spoiled_brat[positive_idx] = torch.min(x_sub_subzero[positive_idx], y_sub_subzero[positive_idx])

                corrected_ans[further_zero] = spoiled_brat
        
        avoided_NaN[zero_idx] = corrected_ans

    return avoided_NaN


def recursive_llr_coset_projection_s_dim(Big_LLR):

    # if first_half.shape[2] == 1:
    #     numerator = 1 + first_half.add(second_half).exp()
    #     denominator =  first_half.exp().add(second_half.exp())
    #     return torch.log(numerator/denominator)
    
    # Big_LLR shape: (batch, 35, 8, 16)
    # Output shape: (batch, 25, 1, 16)

    if Big_LLR.shape[2] == 2:
        # numerator = 1 + Big_LLR.sum(2, keepdim=True).exp()
        # denominator = Big_LLR.exp().sum(2, keepdim=True)
        return log_sum_avoid_zero_NaN(Big_LLR[:, :, 0:1, :], Big_LLR[:, :, 1:2, :])

    else:
        current_coset_length_half = Big_LLR.shape[2] // 2
        first_half = recursive_llr_coset_projection_s_dim(Big_LLR[:, :, :current_coset_length_half, :])
        second_half = recursive_llr_coset_projection_s_dim(Big_LLR[:, :, current_coset_length_half:, :])
        
        # numerator = 1 + first_half.add(second_half).exp()
        # denominator =  first_half.exp().add(second_half.exp())

        return log_sum_avoid_zero_NaN(first_half, second_half)  #(batch, 35, 1, 16) # remember to squeeze the 2nd dimension


def recursive_llr_message_passing_aggregation_s_dim(Big_LLR):

    # Big_LLR shape: (1, 35, 128, 8). -\infty is attached to the first column for each index z.
    # Output shape: (1, 35, 128, 1)
    
    ##############
    ### New implementation using tree.
    ##############

    return -recursive_llr_coset_projection_s_dim(Big_LLR.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    '''
    #######
    ## Old implementation using a chain. This is correct too.
    #######
    
    # if Big_LLR.shape[3] == 2:
    #     numerator = 1 + Big_LLR.sum(3, keepdim=True).exp()
    #     denominator = Big_LLR.exp().sum(3, keepdim=True)
    #     return torch.log(numerator) - torch.log(denominator)
    
    # else:
    #     # print(Big_LLR.shape)
    #     next_friend = Big_LLR[:, :, :, :1]
    #     remaining_Set = recursive_llr_message_passing_aggregation_s_dim(Big_LLR[:, :, :, 1:])

    #     numerator = 1 + next_friend.add(remaining_Set).exp()
    #     denominator =  next_friend.exp().add(remaining_Set.exp())
    #     return torch.log(numerator) - torch.log(denominator) # remember to squeeze the 3nd dimension
    '''


def coset_to_codeword_back_projection_s_dim(decoded_cosets, Coset_idx_unsqueezed):

    # decoded_cosets is of shape (Batch, 28, 64)
    # Coset_idx_unsqueezed is of shape (Batch, 28, 256).
    # Output is of shape (Batch, 28, 256)

    return decoded_cosets.gather(2, Coset_idx_unsqueezed)





####### 2-dim stuff
def fixed_2_subspace_coset_projection_matrix(m, first_non_zero_subspace, second_non_zero_subspace, third_non_zero_subspace):

    M = torch.zeros(2**m, 2**(m-2))

    current_column = 0

    for integer_corresp_coset in range(2**m):

        if sum(M[integer_corresp_coset, :]) == 0:
        
            M[integer_corresp_coset, current_column] = 1

            remaining_element1_in_coset = integer_corresp_coset^first_non_zero_subspace
            remaining_element2_in_coset = integer_corresp_coset^second_non_zero_subspace
            remaining_element3_in_coset = integer_corresp_coset^third_non_zero_subspace

            M[remaining_element1_in_coset, current_column] = 1
            M[remaining_element2_in_coset, current_column] = 1
            M[remaining_element3_in_coset, current_column] = 1

            current_column += 1

    return M


def all_pair_cordinate_directions_2_subspace_coset_projection_tensor(m):

    # We only have (m 2) projections. Input LLR is of shape 2**m. Output LLR is of shape 2**(m-2)
    all_cordinate_2_projections_tensor = torch.zeros(int(m*(m-1)/2), 2**m, 2**(m-2))

    count = 0

    for j in range(1, m):
        for i in range(j):

            all_cordinate_2_projections_tensor[count, :, :] = fixed_2_subspace_coset_projection_matrix(m, 2**i, 2**j, 2**i + 2**j)

            count += 1 

    return all_cordinate_2_projections_tensor


def find_code_projection_2_subspace_coset_indices(M):

    # M is of shape (28, 256, 64). We want a tensor of shape (28, 4, 64) that stores the indices of non-zero elements along each column

    idx = torch.zeros(M.shape[0], 4, M.shape[2]).long()

    for i in range(idx.shape[0]):
        for j in range(idx.shape[2]):
            idx[i, :, j] = (M[i, :, j] == 1.).nonzero().reshape(4)

    return idx


def find_coset_friends_message_passing_and_backprojection(Code_idx):

    # Code_idx is of shape (28, 4, 64). So we want an output of shape (28, 256, 3), where we store the other 3 coset friends for each of the 256 main indices

    LLR_idx = torch.zeros(Code_idx.shape[0], 4*Code_idx.shape[2], 3).long()
    Coset_idx_for_bit_indices = torch.zeros(Code_idx.shape[0], 4*Code_idx.shape[2]).long()

    for i in range(LLR_idx.shape[0]):
        mat = Code_idx[i,:, :]
        for j in range(LLR_idx.shape[1]):
            that_column = (mat == j).nonzero().reshape(2)[1].item()
            all_coset_friends = mat[:, that_column]
            # print(all_coset_friends)
            LLR_idx[i, j, :] = all_coset_friends[all_coset_friends != j]
            Coset_idx_for_bit_indices[i, j] = that_column

    return LLR_idx, Coset_idx_for_bit_indices


def llr_coset_projection_2_subspace_batch(llr, Code_idx): #req_Code_Projection_tensor, numer_req_Code_Projection_tensors, denom_req_Code_Projection_tensors

    # LLR is of shape (1, 256)
    # Code_idx is of shape (28, 4, 64)
    # output is of shape (1, 28, 64)
    Big_LLR = llr[:, Code_idx] # Of shape (1, 28, 4, 64)

    numerator = Big_LLR.sum(2).exp() + Big_LLR[:, :, [0,1], :].sum(2).exp() + Big_LLR[:, :, [0, 2], :].sum(2).exp() + Big_LLR[:, :, [0, 3], :].sum(2).exp() +\
                        Big_LLR[:, :, [1, 2], :].sum(2).exp() + Big_LLR[:, :, [1, 3], :].sum(2).exp() + Big_LLR[:, :, [2, 3], :].sum(2).exp()

    denominator = Big_LLR.exp().sum(2) + Big_LLR[:, :, [0, 1, 2], :].sum(2).exp() + Big_LLR[:, :, [0, 1, 3], :].sum(2).exp() + \
                        Big_LLR[:, :, [0, 2, 3], :].sum(2).exp() + Big_LLR[:, :, [1, 2, 3], :].sum(2).exp()

    return torch.log(1 + numerator) - torch.log(denominator) # (1, 28, 64)


def llr_message_passing_aggregation(llr, LLR_idx):
    
    # llr is of shape (1, 256)
    # LLR_idx is of shape (28, 256, 3)

    Big_LLR = llr[:, LLR_idx] # shape (1, 28, 256, 3)

    numerator = Big_LLR.sum(3).exp() + Big_LLR.exp().sum(3)

    denominator = Big_LLR[:, :, :, [0, 1]].sum(3).exp() + Big_LLR[:, :, :, [0, 2]].sum(3).exp() + Big_LLR[:, :, :, [1, 2]].sum(3).exp()

    return torch.log( numerator ) - torch.log( 1 + denominator) # (1, 28, 256)


def coset_to_codeword_back_projection_2_subspace_batch(decoded_cosets, Coset_idx_unsqueezed):

    # decoded_cosets is of shape (Batch, 28, 64)
    # Coset_idx_unsqueezed is of shape (Batch, 28, 256).
    # Output is of shape (Batch, 28, 256)

    return decoded_cosets.gather(2, Coset_idx_unsqueezed)




#######################################################################################################

def codeword_to_all_coset_projection_and_back(codeword, BatchMul_Code_Projection_tensor):

    # codeword is of shape (1, 256). 
    # Code_Projection_tensor is of shape (255, 256, 256).
    # output is of shape (256, 255)

    return contract('ij, kjm -> ikm', codeword, BatchMul_Code_Projection_tensor).reshape(BatchMul_Code_Projection_tensor.shape[0], BatchMul_Code_Projection_tensor.shape[2]).t() % 2


def codeword_to_all_coset_projection(codeword, Code_Projection_tensor):

    # codeword is of shape (1, 256).
    # Code_Projection_tensor is of shape (255, 256, 128).
    # output is of shape (256, 255)

    return contract('ij, kjm -> ikm', codeword, Code_Projection_tensor).reshape(Code_Projection_tensor.shape[0], Code_Projection_tensor.shape[2]).t() % 2


def coset_to_codeword_back_projection(decoded_cosets, Code_Projection_tensor):

    # decoded_cosets is of shape (255, 128).
    # Code_Projection_tensor is of shape (256, 256, 128). Hence we need to ignore the first component and transpose the last two components
    # output is of shape (256, 255)


    return contract('ij, ijm -> im', decoded_cosets, Code_Projection_tensor[1:, :, :].permute(0, 2, 1)).t() % 2 


def coset_to_codeword_back_projection_batch(decoded_cosets, req_Code_Projection_tensor, projection_choice='no_sparse', proj_indices=None):

    if projection_choice is 'no_sparse': 

    # decoded_cosets is of shape (batch, 255, 128).
    # Code_Projection_tensor is of shape (255, 256, 128). Hence we need to ignore the first component and transpose the last two components
    # output is of shape (batch, 256, 255)

        return contract('kij, ijm -> kim', decoded_cosets, req_Code_Projection_tensor.permute(0, 2, 1)).permute(0, 2, 1) % 2

    elif projection_choice is "static_sparse_proj_batch_wise":

        return contract('kij, kijm -> kim', decoded_cosets, req_Code_Projection_tensor[proj_indices, :, :].permute(0, 1, 3, 2)).permute(0, 2, 1) % 2


def cosetLLR_to_codewordLLR_back_projection_batch(decoded_cosets, req_Code_Projection_tensor, projection_choice='no_sparse', proj_indices=None):

    if projection_choice is 'no_sparse': 

    # decoded_cosets is of shape (batch, 255, 128).
    # Code_Projection_tensor is of shape (255, 256, 128). Hence we need to ignore the first component and transpose the last two components
    # output is of shape (batch, 256, 255)

        return contract('kij, ijm -> kim', decoded_cosets, req_Code_Projection_tensor.permute(0, 2, 1)).permute(0, 2, 1)

    elif projection_choice is "static_sparse_proj_batch_wise":

        return contract('kij, kijm -> kim', decoded_cosets, req_Code_Projection_tensor[proj_indices, :, :].permute(0, 1, 3, 2)).permute(0, 2, 1)


def NNoutput_to_codeword_back_projection_batch(decoded_cosets, Code_Projection_tensor):

    # decoded_cosets is of shape (255, 128).
    # Code_Projection_tensor is of shape (256, 256, 128). Hence we need to ignore the first component and transpose the last two components
    # output is of shape (256, 255)

    return contract('kijl, ijm -> kiml', decoded_cosets, Code_Projection_tensor[1:, :, :].permute(0, 2, 1)).permute(0, 2,1, 3)


def llr_all_coset_projection(llr, req_Code_Projection_tensor):

    # LLR is of shape (1, 256). 
    # Code_Projection_tensor is of shape (255, 256, 128).
    # output is of shape (255, 128)

    exp_llr = torch.exp(llr)

    proj_llr = contract('ij, kjm ->ikm', llr, req_Code_Projection_tensor).reshape(req_Code_Projection_tensor.shape[0],\
                                req_Code_Projection_tensor.shape[2]).t()

    proj_exp_llr = contract('ij, kjm ->ikm', exp_llr, req_Code_Projection_tensor).reshape(req_Code_Projection_tensor.shape[0],\
                                 req_Code_Projection_tensor.shape[2]).t()

    return torch.log(1 + torch.exp(proj_llr)) - torch.log(proj_exp_llr)  


def llr_all_coset_projection_batch(llr, req_Code_Projection_tensor, projection_choice='no_sparse', proj_indices=None):

    # LLR is of shape (1, 256).
    # Code_Projection_tensor is of shape (255, 256, 128).
    # output is of shape (255, 128)

    if projection_choice is "no_sparse":

        exp_llr = torch.exp(llr)

        proj_llr = contract('ij, kjm ->ikm', llr, req_Code_Projection_tensor).reshape(llr.shape[0],req_Code_Projection_tensor.shape[0],\
                                req_Code_Projection_tensor.shape[2]).permute(0,2,1)

        proj_exp_llr = contract('ij, kjm ->ikm', exp_llr, req_Code_Projection_tensor).reshape(llr.shape[0], req_Code_Projection_tensor.shape[0],\
                                 req_Code_Projection_tensor.shape[2]).permute(0,2,1)

        return torch.log(1 + torch.exp(proj_llr)) - torch.log(proj_exp_llr + 1e-8)
    
    elif projection_choice is "static_sparse_proj_batch_wise":

        # proj_indices is of shape (batch_size, proj_indices_for_each_batch). For example, (25, 8)

        modified_Code_Projection_tensor = req_Code_Projection_tensor[proj_indices, :, :] # (25, 8, 256, 128)

        exp_llr = torch.exp(llr)

        proj_llr = contract('ij, ikjm ->ikm', llr, modified_Code_Projection_tensor).reshape(llr.shape[0], modified_Code_Projection_tensor.shape[1],\
                                modified_Code_Projection_tensor.shape[3]).permute(0,2,1)

        proj_exp_llr = contract('ij, ikjm ->ikm', exp_llr, modified_Code_Projection_tensor).reshape(llr.shape[0], modified_Code_Projection_tensor.shape[1],\
                                modified_Code_Projection_tensor.shape[3]).permute(0,2,1)

        return torch.log(1 + torch.exp(proj_llr)) - torch.log(proj_exp_llr)


def permute_a_given_llr(llr, LLR_Permutation_tensor):

    # llr is of shape (1, 256)
    # Permutation_tensor is of shape (255, 256, 256).
    # Output is of shape (256, 255)


    return contract('ij, kjm -> ikm', llr, LLR_Permutation_tensor).reshape(LLR_Permutation_tensor.shape[0], LLR_Permutation_tensor.shape[2]).t()


def permute_a_given_llr_batch(llr, LLR_Permutation_tensor):

    # llr is of shape (1, 256)
    # Permutation_tensor is of shape (255, 256, 256).
    # Output is of shape (256, 255)


    return contract('ij, kjm -> ikm', llr, LLR_Permutation_tensor).reshape(llr.shape[0],LLR_Permutation_tensor.shape[0], LLR_Permutation_tensor.shape[2]).permute(0, 2, 1)


def reed_muller_batch_encoding(batch_messages, Generator_Matrix):

    return batch_messages.mm(Generator_Matrix) % 2


def awgn_channel(batch_codes, snr):

    noise_sigma = snr_db2sigma(snr)

    return (1-2*batch_codes) + noise_sigma*torch.randn(batch_codes.shape[0], batch_codes.shape[1], dtype=batch_codes.dtype)


def simple_awgn_channel(batch_codes, snr, can_I_use_cuda):

    noise_sigma = snr_db2sigma(snr)

    standard_Gaussian = to_var(torch.randn(batch_codes.shape[0], batch_codes.shape[1], dtype=batch_codes.dtype), can_I_use_cuda)

    return batch_codes + noise_sigma*standard_Gaussian


def llr_awgn_channel_affine_Plotkin(corrupted_codewords, snr, a, k_a):

    noise_sigma = snr_db2sigma(snr)

    first_column = (2./noise_sigma**2) * k_a * corrupted_codewords[:, 0:1]

    second_column = (2./noise_sigma**2) * k_a * a * corrupted_codewords[:, 1:2]

    return torch.cat([first_column, second_column], dim=1)

def llr_awgn_channel_bpsk(corrupted_codes, snr):

    noise_sigma = snr_db2sigma(snr)

    return (2./noise_sigma**2) * corrupted_codes


def get_codeword_llr_nn1(code_generator, llr_generator):
    code_data = next(code_generator)
    llr_data = next(llr_generator)
    if code_data.size() != llr_data.size():
        code_data = next(code_generator)
        llr_data = next(llr_generator)
    return code_data, llr_data
