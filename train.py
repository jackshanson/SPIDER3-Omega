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
parser.add_argument('--norm_func', default="layer", type=str, help='Which normalisation function to use: layer, batch, none')
parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
parser.add_argument('--bn_momentum', default=0.99, type=float, help='Batch norm momentum')
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

def Model(input,seq_lens,mask,dropout,num_outputs,is_train,args):
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.BasicLSTMCell
    activation_fn = tf.nn.relu if args.activation.lower() == "relu" else tf.nn.elu
    cell_args = {} 
    cell_args.update({'forget_bias': 1.0})
    layer = [input]
    norm_args = {}
    if args.norm_func.lower() == 'batch':
        norm_func = tf.layers.batch_normalization
        norm_args.update({'training':is_train,'momentum':args.bn_momentum})
    elif args.norm_func.lower() == 'layer':
        norm_func = tf.contrib.layers.layer_norm
    else:
        norm_func = tf.identity
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
                            layer.append(norm_func(layer[-1],**norm_args))
                    layer.append(jack.LSTM_layer(layer[-1],cellfunc,cell_args,rnn_size,seq_lens,False))
                    if args.model_res == True:
                        layer.append(norm_func(layer[-1],**norm_args))
                    else:
                        layer.append(tf.nn.dropout(norm_func(layer[-1],**norm_args),dropout))
                    if args.model_res == True and i!=0:
                        layer.append(layer[-1]+layer[res_start_layer])
        #-------------------CNN 
        elif k == 'CNN':
            filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
            with tf.variable_scope('initCNN'):
                layer.append(tf.layers.conv1d(layer[-1],cnn_size,filter_dims_pattern[0],padding='SAME',activation=None,bias_initializer=tf.constant_initializer(0.01)))
            for i in range(cnn_depth-1):
                if args.model_res == True and i%2 == 0:
                    res_start_layer = len(layer)-1
                with tf.variable_scope('CNN'+str(i+1)):
                    layer.append(norm_func(activation_fn(layer[-1]),**norm_args))
                    cnn_dim = filter_dims_pattern[i%(len(filter_dims_pattern))]
                    layer.append(tf.layers.conv1d(layer[-1],cnn_size,cnn_dim,padding='SAME',activation=None,bias_initializer=tf.constant_initializer(0.01)))
                    if i%2 == 1 and args.model_res:
                        layer.append(layer[-1]+layer[res_start_layer])
            layer.append(norm_func(activation_fn(layer[-1]),**norm_args))
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
    for I,i in enumerate(layer):
        print i.get_shape().as_list(),
        print("%i:"%(I)),
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
        feed_dict = {'oneD_feats:0':data,'seq_lens:0':seq_lens,'ph_dropout:0':1,'mask_bool:0':mask,'output:0':labels,'train_bool:0':False}
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

def test_iter(sess,ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment,metrics,K=0,name=''):
    name = 'Test' if name == '' else name
    print(jack.bcolors.BOLD+name+' set:'+jack.bcolors.RESET)
    tmp_cost,tmp_AUC,tmp_Sw,tmp_MCC,_,tmp_sens,tmp_spec = test_func(sess,ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
    print(name+' AUC = %f'%(tmp_AUC))
    max_Sw_pos = np.argmax(tmp_Sw)
    max_Sw = tmp_Sw[max_Sw_pos]
    max_Sw_thresh = thresholds[0,max_Sw_pos]
    max_MCC_pos = np.nanargmax(tmp_MCC)
    max_MCC = tmp_MCC[max_MCC_pos]
    max_MCC_thresh = thresholds[0,max_MCC_pos]
    print(name+' max Sw = %f, at a threshold of %f'%(max_Sw,max_Sw_thresh))
    print(name+' max MCC = %f, at a threshold of %f'%(max_MCC,max_MCC_thresh))
    metrics['AUC'][K].append(tmp_AUC)
    metrics['cost'][K].append(tmp_cost)
    metrics['Sw'][K].append(tmp_Sw[max_Sw_pos])
    metrics['sens'][K][0].append(tmp_sens[max_Sw_pos])
    metrics['spec'][K][0].append(tmp_spec[max_Sw_pos])
    metrics['MCC'][K].append(tmp_MCC[max_MCC_pos])
    metrics['sens'][K][1].append(tmp_sens[max_MCC_pos])
    metrics['spec'][K][1].append(tmp_spec[max_MCC_pos])
    metrics['thresh'][K][0].append(max_Sw_thresh)
    metrics['thresh'][K][1].append(max_MCC_thresh)
    if np.max(metrics['AUC'][K]) != tmp_AUC:
        print(jack.bcolors.RED+name+' AUC has not increased (%f from epoch %i)'%(np.max(metrics['AUC'][K]),np.argmax(metrics['AUC'][K])+1)+jack.bcolors.RESET)
    else:
        print(jack.bcolors.GREEN+name+' AUC increased!'+jack.bcolors.RESET)
    return metrics


def train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,experiment,num_outputs,num_1D_feats,fpath):
    ph_x = tf.placeholder(tf.float32,[None,None,num_1D_feats],name='oneD_feats')
    ph_seq_lens = tf.placeholder(tf.int32,[None],name='seq_lens')
    ph_dropout = tf.placeholder(tf.float32,name='ph_dropout')
    ph_mask = tf.placeholder(tf.bool,[None,None], name='mask_bool')
    ph_y = tf.placeholder(tf.float32,[None,None,num_outputs],name='output')
    ph_train_bool = tf.placeholder(tf.bool,name='train_bool')
    model = Model(ph_x,ph_seq_lens,ph_mask,ph_dropout,num_outputs,ph_train_bool,args)
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(tf.boolean_mask(ph_y,ph_mask),model[-1],args.gain))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
        batch_size = args.batch_size
        np.random.shuffle(train_ids)
        train_bool = True
        train_save = {'AUC':[[] for i in range(1)],'cost':[[] for i in range(1)],'Sw':[[] for i in range(1)],'MCC':[[] for i in range(1)],'sens':[[[],[]] for i in range(1)],'spec':[[[],[]] for i in range(1)],'thresh':[[[],[]] for i in range(1)]}
        val_save = {'AUC':[[] for i in range(1)],'cost':[[] for i in range(1)],'Sw':[[] for i in range(1)],'MCC':[[] for i in range(1)],'sens':[[[],[]] for i in range(1)],'spec':[[[],[]] for i in range(1)],'thresh':[[[],[]] for i in range(1)]}
        test_save = {'AUC':[[] for i in test_ids],'cost':[[] for i in test_ids],'Sw':[[] for i in test_ids],'MCC':[[] for i in test_ids],'sens':[[[],[]] for i in test_ids],'spec':[[[],[]] for i in test_ids],'thresh':[[[],[]] for i in test_ids]}

        e=1
        step = 0.001
        thresholds = np.arange(0,1+step,step)[None,:]
        while train_bool == True:
            #---------------------TRAINING-------------------------------------#
            train_cost = 0
            print(jack.bcolors.CYAN+'--------------------------Training epoch %i:--------------------------'%(e)+jack.bcolors.RESET)
            for i in tqdm.tqdm(range(int(np.ceil(len(train_ids)/float(batch_size)))),file=sys.stdout):
                train_data,train_labels,mask,seq_lens = get_data(feat_dic,train_ids,batch_size,i,norm_mu,norm_std)
                if experiment == 'PRO':
                    training_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]=='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in train_ids[i*batch_size:np.min([(i+1)*batch_size,len(train_ids)])]]) #proline only
                    mask = np.logical_and(training_mask,mask)
                elif experiment == 'NOPRO':
                    training_mask = np.concatenate([np.concatenate([feat_dic[j][0][None,:]!='P',np.zeros([1,np.max(seq_lens)-feat_dic[j][0].shape[0]])==0],1) for j in train_ids[i*batch_size:np.min([(i+1)*batch_size,len(train_ids)])]]) #proline only
                    mask = np.logical_and(training_mask,mask)
                feed_dict = {'oneD_feats:0':train_data,'seq_lens:0':seq_lens,'ph_dropout:0':0.5,'mask_bool:0':mask,'output:0':train_labels,'train_bool:0':True}
                [_,tmp_cost,_] = sess.run([opt,cost,model],feed_dict=feed_dict)
                train_cost += np.sum(tmp_cost)
            print('Training cost = %f'%(train_cost))
            #----------------------TESTING-------------------------------------#
            print('Testing on train set...')
            train_save = test_iter(sess,train_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment,train_save,K=0,name='Train')

            print('Validating...')
            val_save = test_iter(sess,val_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment,val_save,K=0,name='Validation')
            max_val_epoch = np.argmax(val_save['AUC'])
            if max_val_epoch+1 == e:
                print('Saving model in:'+str(fpath))
                saver.save(sess,fpath)


            for K,k in enumerate(test_ids):
                test_save = test_iter(sess,k,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment,test_save,K=K,name='Test '+str(K))

                print('Best validation epoch results on test data:')
                print('Epoch:\tAUC:\tT:\tSw(T):\tSe(T):\tSp(T):\tT:\tMCC(T):\tSe(T):\tSp(T):')
                
                print('%i\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f'%(max_val_epoch+1,test_save['AUC'][K][max_val_epoch],test_save['thresh'][K][0][max_val_epoch],test_save['Sw'][K][max_val_epoch],test_save['sens'][K][0][max_val_epoch],test_save['spec'][K][0][max_val_epoch],test_save['thresh'][K][1][max_val_epoch],test_save['MCC'][K][max_val_epoch],test_save['sens'][K][1][max_val_epoch],test_save['spec'][K][1][max_val_epoch]))

            if max_val_epoch < e-5:
                print('Stopping training')
                train_bool = False
            e+=1
            #---------------------STOP-----------------------------------------#
    tf.reset_default_graph()
    return [[max_val_epoch+1,test_save['AUC'][K][max_val_epoch],test_save['thresh'][K][0][max_val_epoch],test_save['Sw'][K][max_val_epoch],test_save['sens'][K][0][max_val_epoch],test_save['spec'][K][0][max_val_epoch],test_save['thresh'][K][1][max_val_epoch],test_save['MCC'][K][max_val_epoch],test_save['sens'][K][1][max_val_epoch],test_save['spec'][K][1][max_val_epoch]] for K,k in enumerate(test_ids)]
    
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


#wtf_rhys +/home/rhys/work/tensorflow_code/experiments/github_brnn/experiment_dir/tf-bioinf-brnn/spider3/iteration3-pssm_phys7_hmm30_asa_ttpp_hsea_hseb_cn/160817-19:05:52/results
wtf_rhys = './spd3_outputs'

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
    if i in old_train_ids:
        old_ids = old_train_ids if i in old_train_ids else old_test_ids
        old_db = old_train if i in old_train_ids else old_test
        index = old_ids.index(i)
        seq = old_db['aa'][index]
        feat_dic[i] = [seq]
        feats = np.concatenate([old_db[j][index].astype(float) if j!='spd33' else jack.read_spd33_third_iteration(wtf_rhys+'/combined/'+i+'.spd33',seq) for j in features],1)
        feats[np.isinf(feats)] = 9999 #only HHM, surely...?
        feats[feats>9999] = 9999
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
                    feat[feat>9999] = 9999
                elif j=='phys':
                    feat = jack.read_pccp('',seq,pccp_dic)
                elif j=='spd33':
                    feat=jack.read_spd33_output(database_dir+i+'/'+i+'.spd33',seq)
                feats.append(feat)
            feat_dic[i].append(np.concatenate(feats,1))
        except:
            try:
                del feat_dic[i]
            except:
                pass
            bad_prots.append(i)
            continue
    #---------------------------------------------------------------------------
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
    jack.tee(fpath+'/results_log.txt','%i\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f\n'%tuple(all_res[K]),append=True)
    jack.tee(fpath+'/results_log.txt','Only Proline:\n',append=True)
    jack.tee(fpath+'/results_log.txt','%i\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f\t%1.3f\t%1.4f\t%1.4f\t%1.4f\n'%tuple(pro_res[K]),append=True)
#jack.tee(fpath+'/results_log.txt','No Proline:',append=True)
#jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f'%tuple(nopro_res),append=True)
for arg in vars(args):
    jack.tee(fpath+'/args.txt',"%s:\t%s\n"%(arg,getattr(args,arg)),append=True)

sys.stderr.write('Finished training model in '+fpath)


















