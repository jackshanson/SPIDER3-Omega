import scipy.io as sio
import os,tqdm,warnings
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
#warnings.simplefilter('error')

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
    for i in tqdm.tqdm(range(int(np.ceil(len(ids)/float(batch_size))))):
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
    return running_cost,AUC,Sw,MCC,Q2


def train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,experiment,num_outputs,num_1D_feats,fpath):
    ph_x = tf.placeholder(tf.float32,[None,None,num_1D_feats],name='oneD_feats')
    ph_seq_lens = tf.placeholder(tf.int32,[None],name='seq_lens')
    ph_dropout = tf.placeholder(tf.float32,name='ph_dropout')
    ph_mask = tf.placeholder(tf.bool,[None,None], name='mask_bool')
    ph_y = tf.placeholder(tf.float32,[None,None,num_outputs],name='output')
    model = Model(ph_x,ph_seq_lens,ph_mask,ph_dropout,num_outputs,args)
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(tf.boolean_mask(ph_y,ph_mask),model[-1],args.gain))
    opt = tf.contrib.layers.optimize_loss(cost, None,0.01,optimizer='Adam')
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
        test_AUC_save = []
        train_cost_save = []
        val_cost_save = []
        test_cost_save = []
        train_Sw_save = []
        val_Sw_save = []
        test_Sw_save = []
        train_MCC_save = []
        val_MCC_save = []
        test_MCC_save = []
        train_Q2_save = []
        val_Q2_save = []
        test_Q2_save = []

        test_thresh_save = []
        e=1
        step = 0.01
        thresholds = np.arange(0,1+step,step)[None,:]
        while train_bool == True:
            train_cost = 0
            print(jack.bcolors.CYAN+'Training epoch %i:'%(e)+jack.bcolors.RESET)
            for i in tqdm.tqdm(range(int(np.ceil(len(train_ids)/float(batch_size))))):
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


            print('Testing on train set...')
            train_cost,train_AUC,train_Sw,train_MCC,train_Q2 = test_func(sess,train_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
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
            train_Q2_save.append(train_Q2[max_Sw_pos])
            if np.max(train_AUC_save) != train_AUC:
                print(jack.bcolors.RED+'Train AUC has not increased (%f from epoch %i)'%(np.max(train_AUC_save),np.argmax(train_AUC_save)+1)+jack.bcolors.RESET)
            else:
                print(jack.bcolors.GREEN+'Train AUC increased!'+jack.bcolors.RESET)

            print('Validating...')
            val_cost,val_AUC,val_Sw,val_MCC,val_Q2 = test_func(sess,val_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
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
            val_Q2_save.append(val_Q2[max_Sw_pos])
            if np.max(val_AUC_save) != val_AUC:
                print(jack.bcolors.RED+'Validation AUC has not increased (%f from epoch %i)'%(np.max(val_AUC_save),np.argmax(val_AUC_save)+1)+jack.bcolors.RESET)
            else:
                print(jack.bcolors.GREEN+'Validation AUC increased!'+jack.bcolors.RESET)
                if args.save == True:
                    print(jack.bcolors.GREEN+'Saving in '+fpath+'/model_'+experiment+'.net'+jack.bcolors.RESET)
                    saver.save(sess, fpath+'/model_'+experiment+'.net')


            print('Testing on ALL...')
            test_cost,test_AUC,test_Sw,test_MCC,test_Q2 = test_func(sess,test_ids,batch_size,feat_dic,norm_mu,norm_std,cost,model,thresholds,args,experiment)
            print('Test AUC = %f'%(test_AUC))
            max_Sw_pos = np.argmax(test_Sw)
            max_Sw = test_Sw[max_Sw_pos]
            max_thresh = thresholds[0,max_Sw_pos]
            max_MCC = test_MCC[max_Sw_pos]
            max_Q2 = test_Q2[max_Sw_pos]
            print('Test max Sw = %f, at a threshold of %f'%(max_Sw,max_thresh))
            print('Max MCC at that threshold = %f'%(max_MCC))
            print('Max Q2 at that threshold = %f'%(max_Q2))
            test_AUC_save.append(test_AUC)
            test_cost_save.append(test_cost)
            test_Sw_save.append(test_Sw[max_Sw_pos])
            test_MCC_save.append(test_MCC[max_Sw_pos])
            test_Q2_save.append(test_Q2[max_Sw_pos])
            test_thresh_save.append(max_thresh)
            if np.max(test_AUC_save) != test_AUC:
                print(jack.bcolors.RED+'Test AUC has not increased (%f from epoch %i)'%(np.max(test_AUC_save),np.argmax(test_AUC_save)+1)+jack.bcolors.RESET)
            else:
                print(jack.bcolors.GREEN+'Test AUC increased!'+jack.bcolors.RESET)

        

            max_val_epoch = np.argmax(val_AUC_save)
            print('Best validation epoch results on test data:')
            print('%i\t%f\t%1.2f\t%f\t%f\t%f'%(max_val_epoch+1,test_AUC_save[max_val_epoch],test_thresh_save[max_val_epoch],test_Sw_save[max_val_epoch],test_Q2_save[max_val_epoch],test_MCC_save[max_val_epoch]))

            if max_val_epoch < e-5:
                print('Stopping training')
                train_bool = False
            e+=1
    tf.reset_default_graph()
    return [max_val_epoch+1,test_AUC_save[max_val_epoch],test_thresh_save[max_val_epoch],test_Sw_save[max_val_epoch],test_Q2_save[max_val_epoch],test_MCC_save[max_val_epoch]]
            

wtf_rhys = '/home/rhys/work/tensorflow_code/experiments/github_brnn/experiment_dir/tf-bioinf-brnn/spider3/iteration3-pssm_phys7_hmm30_asa_ttpp_hsea_hseb_cn/160817-19:05:52/results'

features = ['pssm','HHMprob','phys']

train = sio.loadmat('/home/rhys/work/Databases/ourdatabase/combined_train.mat')['combined'][0]
test = sio.loadmat('/home/rhys/work/Databases/ourdatabase/combined_test.mat')['combined'][0]

train_ids = [i[0][:-2] for i in train['name']]
test_ids = [i[0][:-2] for i in test['name']]

all_feats = [np.concatenate([train[i],test[i]]) for i in features]
all_feats = [np.concatenate([all_feats[J][I] for J in range(len(features))],1) for I,i in enumerate(train_ids+test_ids)]
for i in all_feats:
    i[np.isinf(i)] = 9999

all_aa = np.concatenate([train['aa'],test['aa']])

#spider_features
all_spider = [jack.read_spd33(wtf_rhys+'/'+i+'.spd3',all_aa[I]) for I,i in enumerate(train_ids+test_ids)]
all_feats = [np.concatenate([all_feats[I],all_spider[I]],1) for I,i in enumerate(train_ids+test_ids)]

all_labels_raw = np.concatenate([train['omega_raw'],test['omega_raw']])
all_labels = []

for i in all_labels_raw:
    i[np.isnan(i)] = 360
    all_labels.append(np.ones(i.shape)*-1)
    all_labels[-1][np.abs(i)<30] = 1
    all_labels[-1][np.logical_and(np.abs(i)<210,np.abs(i)>150)] = 0


feat_dic = {i:[all_aa[I],all_feats[I],all_labels[I]] for I,i in enumerate(train_ids+test_ids)}


np.random.seed(seed=10)
np.random.shuffle(train_ids)
val_ids = train_ids[int(len(train_ids)*0.9):]
train_ids = train_ids[:int(len(train_ids)*0.9)]

all_train_data = np.concatenate([feat_dic[i][1] for i in train_ids])
norm_mu = np.mean(all_train_data,0)
norm_std = np.std(all_train_data,0)

num_1D_feats = all_train_data.shape[1]
num_outputs = 1

exists_flag = True
model_id = -1
while exists_flag:
    model_id += 1
    fpath = "save_files/model_"+str(model_id)
    exists_flag = os.path.isdir(fpath)
os.makedirs(fpath)

all_res = train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,'ALL',num_outputs,num_1D_feats,fpath)
pro_res = train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,'PRO',num_outputs,num_1D_feats,fpath)
#nopro_res = train_func(feat_dic,train_ids,val_ids,test_ids,norm_mu,norm_std,args,'NOPRO',num_outputs,num_1D_feats)
print(jack.bcolors.BOLD+'\n\nFINAL RESULTS:\n'+jack.bcolors.RESET)
jack.tee(fpath+'/results_log.txt','All residues:\n',append=False)
jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f\n'%tuple(all_res),append=True)
jack.tee(fpath+'/results_log.txt','Only Proline:\n',append=True)
jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f\n'%tuple(pro_res),append=True)
#jack.tee(fpath+'/results_log.txt','No Proline:',append=True)
#jack.tee(fpath+'/results_log.txt','%i\t%f\t%1.2f\t%f\t%f\t%f'%tuple(nopro_res),append=True)
for arg in vars(args):
    jack.tee(fpath+'/args.txt',"%s:\t%s\n"%(arg,getattr(args,arg)),append=(I!=0))




















