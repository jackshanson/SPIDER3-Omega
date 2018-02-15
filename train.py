import scipy.io as sio
import os,tqdm,sys
import argparse
import pandas as pd
import jack_misc as jack
parser = argparse.ArgumentParser()
parser.add_argument('--small', default=False, type=bool, help='Whether or not to run a small train and val set (True or false)')
parser.add_argument('--gpu', default=0, type=int, help='Which GPU to use on the machine')
parser.add_argument('--type', default='ALL', type=str, help='Type of test to run ("ALL","PRO","NOPRO"')
parser.add_argument('--model_res', default=True, type=bool, help='If model is residual or not')
parser.add_argument('--layout', default="CNN", type=str, help='RNN, CNN, or "CNN RNN" etc')
parser.add_argument('--gain', default=100, type=int, help='Loss scaling factor')
parser.add_argument('--RNN_size', default=128, type=int, help='Size of the RNN layer')
parser.add_argument('--CNN_size', default=64, type=int, help='Size of the CNN layer')
parser.add_argument('--RNN_depth', default=2, type=int, help='Number of RNN layers')
parser.add_argument('--CNN_depth', default=20, type=int, help='Number of CNN layers')
parser.add_argument('--FC_size', default=128, type=int, help='Size of the CNN layers')
parser.add_argument('--FC_depth', default=2, type=int, help='Number of FC layers')
parser.add_argument('--activation', default='ELU', type=str, help='Activation function ("ELU","ReLU")')
parser.add_argument('--bottleneck', default=False, type=bool, help='Bottleneck in LSTM layers')
parser.add_argument('--filter_dims', default="3", type=str, help='Pattern for filter kx1 dimensions')
parser.add_argument('--save', default=True, type=bool, help='Whether or not to save this model')
args = parser.parse_args()
AA = 'GALMFWKQESPVICYHRNDT'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
import tensorflow as tf
import numpy as np
import matplotlib,os
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def plot_func(list1,list2,list3,legend=['Train','Val','Test']):
    plt.figure()
    plt.plot(list1)
    plt.plot(list2)
    plt.plot(list3)
    plt.legend(legend)

def read_rhys_omega(fname):
    with open(fname,'r') as f:
        omega = pd.read_csv(f).values
    return omega

def Model(input,seq_lens,mask,dropout,num_outputs,args):
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.BasicLSTMCell
    activation_fn = tf.nn.elu if args.activation == "ELU" else tf.nn.relu
    cell_args = {} 
    cell_args.update({'forget_bias': 1.0})
    layer = [input]
    model_layout = args.layout.split()
    for k in model_layout:
        #-------------------RNN    
        if k == 'RNN':
            for i in range(rnn_depth):
                with tf.variable_scope('RNN'+str(i)):
                    if args.model_res == True:
                        res_start_layer = len(layer)-1
                    if args.model_res == True and args.bottleneck == True:
                        with tf.variable_scope('bottleneck'+str(i)):
                            layer.append(tf.layers.conv1d(layer[-1],rnn_size,1,padding='SAME',activation=activation_fn))
                            layer.append(tf.contrib.layers.layer_norm(layer[-1]))
                    layer.append(jack.LSTM_layer(layer[-1],cellfunc,cell_args,rnn_size,seq_lens,False))
                    if args.model_res == True:
                        layer.append(tf.contrib.layers.layer_norm(layer[-1]))
                    else:
                        layer.append(tf.nn.dropout(tf.contrib.layers.layer_norm(layer[-1]),dropout))
                    if args.model_res == True and i!=0:
                        layer.append(layer[-1]+layer[res_start_layer])
        #-------------------CNN 
        elif k == 'CNN':
            filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
            with tf.variable_scope('initCNN'):
                layer.append(tf.layers.conv1d(layer[-1],cnn_size,filter_dims_pattern[0],padding='SAME',activation=tf.identity))
            for i in range(cnn_depth-1):
                if args.model_res == True and i%2 == 0:
                    res_start_layer = len(layer)-1
                with tf.variable_scope('CNN'+str(i)):
                    layer.append(tf.contrib.layers.layer_norm(activation_fn(layer[-1])))
                    cnn_dim = filter_dims_pattern[i%(len(filter_dims_pattern))]
                    layer.append(tf.layers.conv1d(layer[-1],cnn_size,cnn_dim,padding='SAME',activation=tf.identity))
                    if i%2 == 1 and args.model_res:
                        layer.append(layer[-1]+layer[res_start_layer])
            layer.append(tf.contrib.layers.layer_norm(activation_fn(layer[-1])))
    #-------------------MASK
    layer.append(tf.boolean_mask(layer[-1],mask))
    #-------------------FC
    for i in range(fc_depth):
        with tf.variable_scope('FC'+str(i)):
            layer.append(jack.FC_layer(layer[-1],fc_size,dropout,activation_fn=activation_fn))
    #-------------------OUTPUT
    with tf.variable_scope('Output'):
        layer.append(jack.FC_layer(layer[-1],num_outputs,1,activation_fn=tf.identity))
    #-------------------SUMMARY    
    for i in layer:
        print i.get_shape().as_list(),
        print str(i.name)
    return layer
    
def get_data(feat_dic,ids,batch_size,i,norm_mu,norm_std):
    data = [(feat_dic[j][1]-norm_mu)/norm_std for j in ids[i*batch_size:np.min([(i+1)*batch_size,len(ids)])]]
    seq_lens = [j.shape[0] for j in data]
    max_seq_len = np.max(seq_lens)
    data = np.concatenate([np.concatenate([j,np.zeros([max_seq_len-j.shape[0],j.shape[1]])])[None,:,:] for j in data])
    labels = [feat_dic[j][2] for j in ids[i*batch_size:np.min([(i+1)*batch_size,len(ids)])]]
    labels = np.concatenate([np.concatenate([j,-1*np.ones([max_seq_len-j.shape[0],j.shape[1]])])[None,:,:] for j in labels])
    mask = (labels>-1)[:,:,0]
    return data,labels,mask,seq_lens

def test_func(sess,ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment):
    running_cost = 0
    tp = np.zeros(thresholds.shape[1])
    fp = np.zeros(thresholds.shape[1])
    tn = np.zeros(thresholds.shape[1])
    fn = np.zeros(thresholds.shape[1])
    for i in tqdm.tqdm(range(int(np.ceil(len(ids)/float(batch_size)))),file=sys.stdout):
        data,labels,mask,seq_lens = get_data(feat_dic,ids,batch_size,i,norm_mu,norm_std)
        if experiment == 'PRO':
            seq_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]=='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in ids[i*batch_size:np.min([(i+1)*batch_size,len(ids)])]]) #proline only
            mask = np.logical_and(seq_mask,mask)
        elif experiment == 'NOPRO':
            seq_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]!='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in ids[i*batch_size:np.min([(i+1)*batch_size,len(ids)])]]) #nonproline only
            mask = np.logical_and(seq_mask,mask)
        feed_dict = {'oneD_feats:0':data,'seq_lens:0':seq_lens,'ph_dropout:0':1,'mask_bool:0':mask,'output:0':labels}
        [tmp_cost,tmp_out] = sess.run([cost,model],feed_dict=feed_dict)
        running_cost += np.sum(tmp_cost)
        outputs = jack.sigmoid(tmp_out[-1])
        threshed_outputs = outputs>=thresholds
        reshaped_labels = labels[mask==True]
        tp += np.sum(np.logical_and(threshed_outputs,reshaped_labels),0)
        tn += np.sum(np.logical_and(np.logical_not(threshed_outputs),np.logical_not(reshaped_labels)),0)
        fp += np.sum(np.logical_and(threshed_outputs,np.logical_not(reshaped_labels)),0)
        fn += np.sum(np.logical_and(np.logical_not(threshed_outputs),reshaped_labels),0)
    sens = tp/(tp+fn).astype(float)
    spec = tn/(tn+fp).astype(float)
    AUC = np.trapz(sens,spec)
    Sw = sens+spec-1
    with np.errstate(invalid='ignore'):
        MCC = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp))
    Q2 = (tp+tn)/(tp+tn+fn+fp)
    return running_cost,AUC,Sw,MCC,Q2,sens,spec


def train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,experiment,num_outputs,num_1D_feats,fpath):
    ph_x = tf.placeholder(tf.float32,[None,None,num_1D_feats],name='oneD_feats')
    ph_seq_lens = tf.placeholder(tf.int32,[None],name='seq_lens')
    ph_dropout = tf.placeholder(tf.float32,name='ph_dropout')
    ph_mask = tf.placeholder(tf.bool,[None,None], name='mask_bool')
    ph_y = tf.placeholder(tf.float32,[None,None,num_outputs],name='output')
    model = Model(ph_x,ph_seq_lens,ph_mask,ph_dropout,num_outputs,args)
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(tf.boolean_mask(ph_y,ph_mask),model[-1],args.gain))
    opt = tf.contrib.layers.optimize_loss(cost, None,0.01,optimizer='Adam')

    if args.save == True:    
        saver = tf.train.Saver() 

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    config.log_device_placement=False
    with tf.Session(config=config) as sess:
        sess.run(init)
        batch_size = 50
        np.random.shuffle(train_ids)
        train_bool = True
        train_AUC_save = []
        val_AUC_save = []
        test_AUC_save = [[] for i in test_ids]
        train_cost_save = []
        val_cost_save = []
        test_cost_save = [[] for i in test_ids]
        train_Sw_save = []
        val_Sw_save = []
        test_Sw_save = [[] for i in test_ids]
        train_MCC_save = []
        val_MCC_save = []
        test_MCC_save = [[] for i in test_ids]
        train_Q2_save = []
        val_Q2_save = []
        test_Q2_save = [[] for i in test_ids]
        train_sens_save = []
        val_sens_save = []
        test_sens_save = [[] for i in test_ids]
        train_spec_save = []
        val_spec_save = []
        test_spec_save = [[] for i in test_ids]

        test_thresh_save = [[] for i in test_ids]
        e=1
        step = 0.01
        thresholds = np.arange(0,1+step,step)[None,:]
        while train_bool == True:
            train_cost = 0
            print(jack.bcolors.CYAN+'Training epoch %i:'%(e)+jack.bcolors.RESET)
            for i in tqdm.tqdm(range(int(np.ceil(len(train_ids)/float(batch_size)))),file=sys.stdout):
                train_data,train_labels,mask,seq_lens = get_data(feat_dic,train_ids,batch_size,i,norm_mu,norm_std)
                if experiment == 'PRO':
                    training_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]=='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in train_ids[i*batch_size:np.min([(i+1)*batch_size,len(train_ids)])]]) #proline only
                    mask = np.logical_and(training_mask,mask)
                elif experiment == 'NOPRO':
                    training_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]!='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in train_ids[i*batch_size:np.min([(i+1)*batch_size,len(train_ids)])]]) #proline only
                    mask = np.logical_and(training_mask,mask)
                feed_dict = {'oneD_feats:0':train_data,'seq_lens:0':seq_lens,'ph_dropout:0':0.5,'mask_bool:0':mask,'output:0':train_labels}
                [_,tmp_cost,_] = sess.run([opt,cost,model],feed_dict=feed_dict)
                train_cost += np.sum(tmp_cost)
            print('Training cost = %f'%(train_cost))

            print('Testing on train set...')
            train_cost,train_AUC,train_Sw,train_MCC,train_Q2,train_sens,train_spec = test_func(sess,train_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
            print('Train AUC = %f'%(train_AUC))
            max_Sw_pos = np.argmax(train_Sw)
            max_Sw = train_Sw[max_Sw_pos]
            max_thresh = thresholds[0,max_Sw_pos]
            max_MCC = train_MCC[max_Sw_pos]
            max_Q2 = train_Q2[max_Sw_pos]
            print('Train max Sw = %f, at a threshold of %f'%(max_Sw,max_thresh))
            print('Max MCC at that threshold = %f'%(max_MCC))
            print('Max Q2 at that threshold = %f'%(max_Q2))
            train_AUC_save.append(train_AUC)
            train_cost_save.append(train_cost)
            train_Sw_save.append(train_Sw[max_Sw_pos])
            train_MCC_save.append(train_MCC[max_Sw_pos])
            train_sens_save.append(train_sens[max_Sw_pos])
            train_spec_save.append(train_spec[max_Sw_pos])
            train_Q2_save.append(train_Q2[max_Sw_pos])
            if np.max(train_AUC_save) != train_AUC:
                print(jack.bcolors.RED+'Train AUC has not increased (%f from epoch %i)'%(np.max(train_AUC_save),np.argmax(train_AUC_save)+1)+jack.bcolors.RESET)
            else:
                print(jack.bcolors.GREEN+'Train AUC increased!'+jack.bcolors.RESET)

            print('Validating...')
            val_cost,val_AUC,val_Sw,val_MCC,val_Q2,val_sens,val_spec = test_func(sess,val_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
            print('Validation AUC = %f'%(val_AUC))
            max_Sw_pos = np.argmax(val_Sw)
            max_Sw = val_Sw[max_Sw_pos]
            max_thresh = thresholds[0,max_Sw_pos]
            max_MCC = val_MCC[max_Sw_pos]
            max_Q2 = val_Q2[max_Sw_pos]
            print('Validation max Sw = %f, at a threshold of %f'%(max_Sw,max_thresh))
            print('Max MCC at that threshold = %f'%(max_MCC))
            print('Max Q2 at that threshold = %f'%(max_Q2))
            val_AUC_save.append(val_AUC)
            val_cost_save.append(val_cost)
            val_Sw_save.append(val_Sw[max_Sw_pos])
            val_MCC_save.append(val_MCC[max_Sw_pos])
            val_sens_save.append(val_sens[max_Sw_pos])
            val_spec_save.append(val_spec[max_Sw_pos])
            val_Q2_save.append(val_Q2[max_Sw_pos])
            if np.max(val_AUC_save) != val_AUC:
                print(jack.bcolors.RED+'Validation AUC has not increased (%f from epoch %i)'%(np.max(val_AUC_save),np.argmax(val_AUC_save)+1)+jack.bcolors.RESET)
            else:
                print(jack.bcolors.GREEN+'Validation AUC increased!'+jack.bcolors.RESET)
                if args.save == True:
                    print(jack.bcolors.GREEN+'Saving in '+fpath+'/model_'+experiment+'.net'+jack.bcolors.RESET)
                    if args.save == True:
                        saver.save(sess, fpath+'/model_'+experiment+'.net')
            max_val_epoch = np.argmax(val_AUC_save)


            for K,k in enumerate(test_ids):
                print(jack.bcolors.BOLD+'Test set '+str(K)+jack.bcolors.RESET)
                test_cost,test_AUC,test_Sw,test_MCC,test_Q2,test_sens,test_spec = test_func(sess,k,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
                print('Test AUC = %f'%(test_AUC))
                max_Sw_pos = np.argmax(test_Sw)
                max_Sw = test_Sw[max_Sw_pos]
                max_thresh = thresholds[0,max_Sw_pos]
                max_MCC = test_MCC[max_Sw_pos]
                max_Q2 = test_Q2[max_Sw_pos]
                print('Test max Sw = %f, at a threshold of %f'%(max_Sw,max_thresh))
                print('Max MCC at that threshold = %f'%(max_MCC))
                print('Max Q2 at that threshold = %f'%(max_Q2))
                test_AUC_save[K].append(test_AUC)
                test_cost_save[K].append(test_cost)
                test_Sw_save[K].append(test_Sw[max_Sw_pos])
                test_MCC_save[K].append(test_MCC[max_Sw_pos])
                test_Q2_save[K].append(test_Q2[max_Sw_pos])
                test_sens_save[K].append(test_sens[max_Sw_pos])
                test_spec_save[K].append(test_spec[max_Sw_pos])
                test_thresh_save[K].append(max_thresh)
                if np.max(test_AUC_save[K]) != test_AUC:
                    print(jack.bcolors.RED+'Test AUC has not increased (%f from epoch %i)'%(np.max(test_AUC_save[K]),np.argmax(test_AUC_save[K])+1)+jack.bcolors.RESET)
                else:
                    print(jack.bcolors.GREEN+'Test AUC increased!'+jack.bcolors.RESET)

                print('Best validation epoch results on test data:')
                print('Epoch:\tAUC:\tT:\tSw(T):\tQ2(T):\tMCC(T):\tSens(T):\tSpec(T):')
                print('%i\t\t%f\t%1.2f\t%f\t%f\t%f\t%f\t\t%f'%(max_val_epoch+1,test_AUC_save[K][max_val_epoch],test_thresh_save[K][max_val_epoch],test_Sw_save[K][max_val_epoch],test_Q2_save[K][max_val_epoch],test_MCC_save[K][max_val_epoch],test_sens_save[K][max_val_epoch],test_spec_save[K][max_val_epoch]))

            if max_val_epoch < e-5:
                print('Stopping training')
                train_bool = False
            e+=1
    tf.reset_default_graph()
    return [[max_val_epoch+1,test_AUC_save[K][max_val_epoch],test_thresh_save[K][max_val_epoch],test_Sw_save[K][max_val_epoch],test_Q2_save[K][max_val_epoch],test_MCC_save[K][max_val_epoch],test_sens_save[K][max_val_epoch],test_spec_save[K][max_val_epoch]] for K,k in enumerate(test_ids)]
    
pc_name = os.uname()[1]        
data_dir = 'Protein/' if pc_name == 'JackPC' else ''

with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_train_spot_contact') as f:
    train_ids = f.read().splitlines()
    train_ids = train_ids[:200] if args.small else train_ids
with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_val_spot_contact') as f:
    val_ids = f.read().splitlines()
    val_ids = val_ids[:50] if args.small else val_ids
with open('/home/jack/Documents/Databases/Contact_Map/dat/spot-contact-lists/list_indtest_spot_contact') as f:
    test_ids = f.read().splitlines()
with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_indtest_E0.1') as f:
    e01_ids = f.read().splitlines()



wtf_rhys = '/home/rhys/work/tensorflow_code/experiments/github_brnn/experiment_dir/tf-bioinf-brnn/spider3/iteration3-pssm_phys7_hmm30_asa_ttpp_hsea_hseb_cn/160817-19:05:52/results'

old_train = sio.loadmat('/home/rhys/work/Databases/ourdatabase/combined_train.mat')['combined'][0]
old_test = sio.loadmat('/home/rhys/work/Databases/ourdatabase/combined_test.mat')['combined'][0]

old_train_ids = [i[0][:-2] for i in old_train['name']]
old_test_ids = [i[0][:-2] for i in old_test['name']]

database_dir = '/home/jack/Documents/Databases/'+data_dir+'Contact_Map/proteins/'

labels = []
feat_dic  = {}
pccp_dic = jack.get_pccp_dic('/home/jack/Documents/Databases/aa_phy7')
bad = 0
bad_prots = []
features = ['pssm','HHMprob','phys','spd33']

all_ids = pd.unique(train_ids + test_ids + e01_ids + val_ids)# + old_test_ids)

for I,i in enumerate(tqdm.tqdm(all_ids,file=sys.stdout)):
    if i in bad_prots:
        continue
    #---------------------------------------------------------------------------
    if i in old_train_ids or i in old_test_ids:
        old_ids = old_train_ids if i in old_train_ids else old_test_ids
        old_db = old_train if i in old_train_ids else old_test
        index = old_ids.index(i)
        seq = old_db['aa'][index]
        feat_dic[i] = [seq]
        feats = np.concatenate([old_db[j][index].astype(float) if j!='spd33' else jack.read_spd33_features(wtf_rhys+'/'+i+'.spd3',seq) for j in features],1)
        feats[np.isinf(feats)] = 9999 #only HHM, surely...?
        feat_dic[i].append(feats)
        omega_raw = old_db['omega_raw'][index]
        #---------------------------------------------------------------------------
    else:
        try:
            seq, omega_raw = jack.read_omega_file(database_dir+i+'/'+i+'.omegalab')
            feat_dic[i] = [np.array([j for j in seq])]
            feats = []
            omega_raw = omega_raw[:,None]
            for j in features:
                if j=='pssm':
                    feat = jack.read_pssm(database_dir+i+'/'+i+'.pssm',seq)
                elif j=='HHMprob':
                    feat = jack.read_hhm(database_dir+i+'/'+i+'.hhm',seq)
                elif j=='phys':
                    feat = jack.read_pccp('',seq,pccp_dic)
                elif j=='spd33':
                    feat=jack.read_spd33_output(database_dir+i+'/'+i+'.spd33',seq)
                feats.append(feat)
            feat_dic[i].append(np.concatenate(feats,1))
        #---------------------------------------------------------------------------
        except:
            try:
                del feat_dic[i]
            except:
                pass
            bad_prots.append(i)
            continue
    omega_raw[np.isnan(omega_raw)] = 360
    omega_labels = np.ones(omega_raw.shape)*-1    
    omega_labels[np.abs(omega_raw)<30] = 1
    omega_labels[np.logical_and(np.abs(omega_raw)<210,np.abs(omega_raw)>150)] = 0
    feat_dic[i].append(omega_labels)
#train_ids = [i for i in train_ids if i not in old_test_ids]
train_ids = [i for i in train_ids if i not in bad_prots]
val_ids = [i for i in val_ids if i not in bad_prots]
test_ids = [i for i in test_ids if i not in bad_prots]
#old_test_ids = [i for i in old_test_ids if i not in bad_prots]
e01_ids = [i for i in e01_ids if i not in bad_prots]

'''all_feats = [np.concatenate([train[i],test[i]]) for i in features]
all_feats = [np.concatenate([all_feats[J][I] for J in range(len(features))],1) for I,i in enumerate(train_ids+test_ids)]
for i in all_feats:
    i[np.isinf(i)] = 9999
all_aa = np.concatenate([train['aa'],test['aa']])

#spider_features
all_spider = [jack.read_spd33_features(wtf_rhys+'/'+i+'.spd3',all_aa[I]) for I,i in enumerate(train_ids+test_ids)]
all_feats = [np.concatenate([all_feats[I],all_spider[I]],1) for I,i in enumerate(train_ids+test_ids)]

all_labels_raw = np.concatenate([train['omega_raw'],test['omega_raw']])
all_labels = []

for i in all_labels_raw:
    i[np.isnan(i)] = 360
    all_labels.append(np.ones(i.shape)*-1)
    all_labels[-1][np.abs(i)<30] = 1
    all_labels[-1][np.logical_and(np.abs(i)<210,np.abs(i)>150)] = 0

hey = hi

feat_dic = {i:[all_aa[I],all_feats[I],all_labels[I]] for I,i in enumerate(train_ids+test_ids)}


np.random.seed(seed=10)
np.random.shuffle(train_ids)
val_ids = train_ids[int(len(train_ids)*0.9):]
train_ids = train_ids[:int(len(train_ids)*0.9)]'''

all_train_data = np.concatenate([feat_dic[i][1] for i in train_ids])
norm_mu = np.mean(all_train_data,0)
norm_std = np.std(all_train_data,0)

num_1D_feats = all_train_data.shape[1]
num_outputs = 1
fpath = ''
#hey = hi

exists_flag = True
model_id = -1
while exists_flag:
    model_id += 1
    fpath = "save_files_large/model_"+str(model_id)
    exists_flag = os.path.isdir(fpath)
if args.save == True:
    os.makedirs(fpath)
all_res = train_func(feat_dic,train_ids,val_ids,[test_ids,e01_ids],norm_mu,norm_std,args,'ALL',num_outputs,num_1D_feats,fpath)
pro_res = train_func(feat_dic,train_ids,val_ids,[test_ids,e01_ids],norm_mu,norm_std,args,'PRO',num_outputs,num_1D_feats,fpath)
#nopro_res = train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,'NOPRO',num_outputs,num_1D_feats)
print(jack.bcolors.BOLD+'\n\nFINAL RESULTS:\n'+jack.bcolors.RESET)
for K,k in enumerate([test_ids,e01_ids]):
    jack.tee(fpath+'/results_log.txt','python %s\n'%(' '.join(sys.argv)),append=True)
    jack.tee(fpath+'/results_log.txt','All residues:\n',append=True)
    jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f\t%f\t%f\n'%tuple(all_res[K]),append=True)
    jack.tee(fpath+'/results_log.txt','Only Proline:\n',append=True)
    jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f\t%f\t%f\n'%tuple(pro_res[K]),append=True)
#jack.tee(fpath+'/results_log.txt','No Proline:',append=True)
#jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f'%tuple(nopro_res),append=True)
for arg in vars(args):
    jack.tee(fpath+'/args.txt',"%s:\t%s\n"%(arg,getattr(args,arg)),append=(I!=0))

sys.stderr.write('Finished training model in '+fpath)


















