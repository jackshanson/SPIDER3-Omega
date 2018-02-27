import numpy as np
import tensorflow as tf
import pandas as pd

class bcolors:
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"

#-------------------------------------------------------------------------------
#
#       FILE PROCESSING
#
#-------------------------------------------------------------------------------
def get_pccp_dic(fname):
    with open(fname,'r') as f:
        pccp = f.read().splitlines()
        pccp = [i.split() for i in pccp]
        pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}
    return pccp_dic

def read_pccp(fname,seq,pccp_dic):
    return np.array([pccp_dic[i] for i in seq])

def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm

def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)    

def spd3_feature_sincos(x,seq,norm_ASA=True):
    ASA = x[:,0]
    if norm_ASA==True:
        rnam1_std = "ACDEFGHIKLMNPQRSTVWY-"
        ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                            185, 160, 145, 180, 225, 115, 140, 155, 255, 230,1)
        dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
        ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
        ASA = (ASA/ASA_div)[:,None]
    else:
        ASA = ASA[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,7:10]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)

def read_spd33_output(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return tmp_spd3

def read_spd33_third_iteration(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True,skiprows=1,header=None).values[:,1:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq,norm_ASA=False)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return tmp_spd3

def read_spd33_features(fnameclass,fnamereg,seq):
    with open(fnameclass,'r') as f:
        spd3_class = pd.read_csv(f,delim_whitespace=True,header=None,skiprows=1).values.astype(float)
    with open(fnamereg,'r') as f:
        spd3_reg = pd.read_csv(f,delim_whitespace=True,header=None,skiprows=1).values.astype(float)[:,:-3]
    theta_tau_phi_psi = spd3_reg[:,1:9]*2-1 #convert sigmoided outputs to sin/cos outputs
    #switch tt and pp around
    spd3_reg[:,1:9] = np.concatenate([theta_tau_phi_psi[:,4:],theta_tau_phi_psi[:,:4]],1)
    spd3_reg[:,9] *= 50 
    spd3_reg[:,10] *= 65 
    spd3_features = np.concatenate([spd3_reg,spd3_class],1)
    if spd3_features.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return spd3_features

def tee(fname,in_str,append=True):
    fperm = 'a' if append==True else 'w'
    with open(fname,fperm) as f:
        print(in_str),  
        f.write(in_str)

def read_omega_file(fname):
    with open(fname,'r') as f:
        contents = f.read().splitlines()
    AA = contents[1]
    omega = np.array(contents[3].split()).astype(float)
    return AA, omega

#-------------------------------------------------------------------------------
#
#       MODEL FUNCTIONS
#
#-------------------------------------------------------------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x,axis=1):
    return np.exp(x)/(np.sum(np.exp(x),1)[:,None])

def LSTM_layer(input,cell,cell_args,cellsize,seq_lens,time_major):
    cells = [cell(cellsize,**cell_args),cell(cellsize,**cell_args)]
    (fw,bw),_ = tf.nn.bidirectional_dynamic_rnn(cells[0],cells[1],input,sequence_length=seq_lens,dtype=tf.float32,swap_memory=True,time_major=time_major)
    return tf.concat([fw,bw],2)

def FC_layer(input,netsize,dropout,activation_fn=tf.nn.relu):
    W = tf.get_variable('W',shape=[input.get_shape().as_list()[-1],netsize],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    b = tf.get_variable('b',shape=[netsize],initializer=tf.constant_initializer(0.01))
    output = tf.nn.dropout(activation_fn(tf.matmul(input,W)+b),dropout)
    return output

#-------------------------------------------------------------------------------
#
#       RESULTS ANALYSIS
#
#-------------------------------------------------------------------------------

def sensitivity(tp,tn,fp,fn):
    return tp/(tp+fn).astype(float)

def specificity(tp,tn,fp,fn):
    return tn/(tn+fp).astype(float)

def precision(tp,tn,fp,fn):
    with np.errstate(invalid='ignore'):
        return tp/(tp+fp).astype(float)

def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fn+fp).astype(float)

def AUC(sens,spec):
    return np.trapz(sens,spec)

def Sw(tp,tn,fp,fn):
    return sensitivity(tp,tn,fp,fn) + specificity(tp,tn,fp,fn) - 1

def MCC(tp,tn,fp,fn):
    with np.errstate(invalid='ignore'):
        return ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp).astype(np.float64))
