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

def read_spd33(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True,header=None,skiprows=1).values.astype(float)
    if spd3_features.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return spd3_features

#-------------------------------------------------------------------------------
#
#       MODEL FUNCTIONS
#
#-------------------------------------------------------------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))

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

def AUC(sens,spec):
    return np.trapz(sens,spec)

def Sw(tp,tn,fp,fn):
    return sensitivity(tp,tn,fp,fn) + specificity(tp,tn,fp,fn) - 1

def MCC(tp,tn,fp,fn):
    with np.errstate(invalid='ignore'):
        return ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp))
