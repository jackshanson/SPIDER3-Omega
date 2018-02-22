import scipy.io as sio
import os,tqdm,sys,glob
import cPickle as pickle
import pandas as pd
import jack_misc as jack
import matplotlib,os
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


AA_names = 'ACDEFGHIKLMNPQRSTVWY'

import numpy as np
import matplotlib,os
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

pc_name = os.uname()[1]        
data_dir = 'Protein/' if pc_name == 'JackPC' else ''
database_dir = '/home/jack/Documents/Databases/'+data_dir+'Contact_Map/proteins/'

dir_list = glob.glob('./save_files_large/*')
id = [int(i[-2:].strip('_')) for i in dir_list]
id,dir_list = zip(*sorted(zip(id, dir_list)))
dir_list = list(dir_list)
id = list(id)

#dir_list = ['']
#id = ['']
data = {}
for I,i in enumerate(dir_list[::-1]):
    try:
        with open(i+'/test_outputs.p','r') as f:
            data[i] = pickle.load(f)
    except:
        index = dir_list.index(i)
        del dir_list[index]
        del id[index]
        pass

print('Data loaded!')

with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_val_spot_contact') as f:
    val_ids = f.read().splitlines()
with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_indtest_spot_contact') as f:
    test_ids = f.read().splitlines()
with open('/home/jack/Documents/Databases/'+data_dir+'Contact_Map/dat/spot-contact-lists/list_indtest_E0.1') as f:
    e01_ids = f.read().splitlines()

with open('/home/jack/Documents/Databases/Contact_Map/list_neff','r') as f:
    neff_list = f.read().split()

neff_dic = {neff_list[I]:float(neff_list[I+1]) for I in range(0,len(neff_list),2)}

name_datasets = ['Validation','Test','Test-Hard']
dataset_ids = [val_ids,test_ids,e01_ids]

experiments = ['ALL','PRO']
#ROC Analysis
sens = [[[] for _ in name_datasets ] for _ in experiments]
spec = [[[] for _ in name_datasets ] for _ in experiments]
step = 0.001
thresholds = np.arange(0,1+step,step)[None,:]
bad_data = []

thresh_max_MCC = []
for I,i in enumerate(dir_list):  
    print(jack.bcolors.CYAN+'Data source: %s'%(i)+jack.bcolors.RESET)
    print(jack.bcolors.CYAN+'Python command: %s'%(data[i]['cmd'])+jack.bcolors.RESET)
    for M,m in enumerate(experiments):
        print(jack.bcolors.BOLD+'Experiment %s:'%(m)+jack.bcolors.RESET)
        for K,k in enumerate(name_datasets):  
            count = 0
            tp = np.zeros(thresholds.shape)
            fp = np.zeros(thresholds.shape)
            fn = np.zeros(thresholds.shape)
            tn = np.zeros(thresholds.shape)
            for J,j in enumerate(dataset_ids[K]):
                try:
                    threshed_results = np.greater_equal(data[i][m][j]['outputs'],thresholds)
                    labels = data[i][m][j]['masked_labels']
                except:
                    continue
                if threshed_results.shape[0]!=labels.shape[0]:
                    continue
                count+=1
                tp += np.sum(np.logical_and(threshed_results,labels),0)
                tn += np.sum(np.logical_and(np.logical_not(threshed_results),np.logical_not(labels)),0)
                fp += np.sum(np.logical_and(threshed_results,np.logical_not(labels)),0)
                fn += np.sum(np.logical_and(np.logical_not(threshed_results),labels),0) 
            print('Total proteins analysed: %i'%(count))
            if count > 0:
                sens[M][K].append((tp/(tp+fn).astype(float)).squeeze())
                spec[M][K].append((tn/(tn+fp).astype(float)).squeeze())
                AUC = np.trapz(sens[M][K][-1],spec[M][K][-1])
                print('AUC of %s:'%(k)),
                print(AUC)
                if k=='Validation':
                    MCC = jack.MCC(tp,tn,fp,fn)
                    thresh_max_MCC.append(np.nanargmax(MCC))
            else:
                print("%s has nothing good for dataset %s"%(i,k))
                bad_data.append([i,k])
    print('---------------------------------------------------')

sio.savemat('./evaluation/AUC_plots.mat',{'sensitivity':sens,'specificity':spec,'ids':dir_list})
'''
dataset_ind = name_datasets.index('Test')
experiment_ind = experiments.index('ALL')
plt.plot(1-np.array(spec[experiment_ind][dataset_ind]).T,np.array(sens[experiment_ind][dataset_ind]).T)
'''
sens_AA = [[] for _ in name_datasets]
spec_AA = [[] for _ in name_datasets]
prec_AA = [[] for _ in name_datasets]

#----------------------------------Per AA analysis------------------------------
key = data.keys()[0]
count_AA = np.zeros([len(AA_names),len(name_datasets)])
for K,k in enumerate(name_datasets):
    for J,j in enumerate(dataset_ids[K]):
        try:
            seq = data[key]['ALL'][j]['seq'][data[key]['ALL'][j]['labels'][:,0]!=-1]
        except:
            continue
        count_AA[:,K] += np.array([np.sum(np.logical_and(seq==m, data[key]['ALL'][j]['labels'][data[key]['ALL'][j]['labels'][:,0]!=-1].squeeze()==1)) for m in AA_names])
        


print('Residue Analysis:'+jack.bcolors.RESET)
for I,i in enumerate(dir_list):  
    print(jack.bcolors.CYAN+'Data source: %s'%(i)+jack.bcolors.RESET)
    print(jack.bcolors.CYAN+'Python command: %s'%(data[i]['cmd'])+jack.bcolors.RESET)
    for K,k in enumerate(name_datasets):
        AA = np.array([z for z in AA_names])[count_AA[:,K]>0]  
        count = 0
        tp_AA = np.zeros([len(AA),thresholds.shape[1]])
        fp_AA  = np.zeros([len(AA),thresholds.shape[1]])
        fn_AA  = np.zeros([len(AA),thresholds.shape[1]])
        tn_AA  = np.zeros([len(AA),thresholds.shape[1]])
        for J,j in enumerate(dataset_ids[K]):
            try:
                threshed_results = np.greater_equal(data[i]['ALL'][j]['outputs'],thresholds)
                labels = data[i]['ALL'][j]['masked_labels']
            except:
                continue
            if threshed_results.shape[0]!=labels.shape[0]:
                continue
            count+=1
            seq = data[i]['ALL'][j]['seq'][data[i]['ALL'][j]['labels'][:,0]!=-1]
            for R,res in enumerate(AA):
                inds = seq==res
                if inds.sum() == 0: continue
                tp_AA[R,:] += np.sum(np.logical_and(threshed_results[inds],labels[inds]),0)
                tn_AA[R,:] += np.sum(np.logical_and(np.logical_not(threshed_results[inds]),np.logical_not(labels[inds])),0)
                fp_AA[R,:] += np.sum(np.logical_and(threshed_results[inds],np.logical_not(labels[inds])),0)
                fn_AA[R,:] += np.sum(np.logical_and(np.logical_not(threshed_results[inds]),labels[inds]),0) 
        print('Total proteins analysed: %i'%(count))
        sens_AA[K].append((tp_AA[:,thresh_max_MCC[I*len(experiments)]]/(tp_AA[:,thresh_max_MCC[I*len(experiments)]]+fn_AA[:,thresh_max_MCC[I*len(experiments)]]).astype(float)).squeeze())
        spec_AA[K].append((tn_AA[:,thresh_max_MCC[I*len(experiments)]]/(tn_AA[:,thresh_max_MCC[I*len(experiments)]]+fp_AA[:,thresh_max_MCC[I*len(experiments)]]).astype(float)).squeeze())
        prec_AA[K].append((tp_AA[:,thresh_max_MCC[I*len(experiments)]]/(tp_AA[:,thresh_max_MCC[I*len(experiments)]]+fp_AA[:,thresh_max_MCC[I*len(experiments)]]).astype(float)).squeeze())
        print('AA\t'+'\t'.join(AA))
        print('Count&\t'+'%3i&\t'*len(AA)%(tuple(count_AA[:,K][count_AA[:,K]>0])))
        print('Sens\t'+'%1.3f\t'*len(AA)%(tuple(sens_AA[K][-1])))
        print('Spec\t'+'%1.3f\t'*len(AA)%(tuple(spec_AA[K][-1])))
        print('Prec\t'+'%1.3f\t'*len(AA)%(tuple(prec_AA[K][-1])))
        #AUC = np.trapz(sens[M][K][-1],spec[M][K][-1])
    print('---------------------------------------------------')

sio.savemat('./evaluation/per_residue.mat',{'sensitivity_AA':sens_AA,'specificity_AA':spec_AA,'precision_AA':prec_AA,'AA_count':count_AA,'ids':dir_list})



#----------------------------------Neff analysis--------------------------------
MCC_neff = [[] for _ in name_datasets]
max_neff = 12
print('%i\t'*max_neff%(tuple(range(1,max_neff+1))))
for I,i in enumerate(name_datasets):
    neff_count = np.zeros([max_neff])
    for J,j in enumerate(dataset_ids[I]):
        try:
            labels=data[data.keys()[0]]['ALL'][j]['masked_labels']
            label_true_count = np.sum(labels==1)
            neff = int(np.round(neff_dic[j]))-1
            neff_count[neff] += label_true_count
        except:
            pass
    print('%i\t'*max_neff%(tuple(neff_count)))

for I,i in enumerate(dir_list):  
    print(jack.bcolors.CYAN+'Data source: %s'%(i)+jack.bcolors.RESET)
    print(jack.bcolors.CYAN+'Python command: %s'%(data[i]['cmd'])+jack.bcolors.RESET)
    for K,k in enumerate(name_datasets):
        count = 0
        tp_neff = np.zeros([max_neff,thresholds.shape[1]])
        fp_neff  = np.zeros([max_neff,thresholds.shape[1]])
        fn_neff  = np.zeros([max_neff,thresholds.shape[1]])
        tn_neff  = np.zeros([max_neff,thresholds.shape[1]])
        for J,j in enumerate(dataset_ids[K]):
            try:
                threshed_results = np.greater_equal(data[i]['ALL'][j]['outputs'],thresholds)
                labels = data[i]['ALL'][j]['masked_labels']
            except:
                continue
            if threshed_results.shape[0]!=labels.shape[0]:
                continue
            count+=1
            neff = int(np.round(neff_dic[j]))-1
            if neff >11: continue
            tp_neff[neff:,:] += np.sum(np.logical_and(threshed_results,labels),0)
            tn_neff[neff:,:] += np.sum(np.logical_and(np.logical_not(threshed_results),np.logical_not(labels)),0)
            fp_neff[neff:,:] += np.sum(np.logical_and(threshed_results,np.logical_not(labels)),0)
            fn_neff[neff:,:] += np.sum(np.logical_and(np.logical_not(threshed_results),labels),0) 
        MCC_neff[K].append(jack.MCC(tp_neff,tn_neff,fp_neff,fn_neff)[:,thresh_max_MCC[I*len(experiments)]])
        print('Neff\t'+'%4i\t'*max_neff%(tuple(range(1,max_neff+1))))
        print('MCC\t'+'%1.3f\t'*max_neff%(tuple(MCC_neff[K][-1])))



sio.savemat('./evaluation/per_neff.mat',{'MCC_Neff':MCC_neff,'ids':dir_list})



#----------------------------------Excel plotting-------------------------------
metrics = {}
for I,i in enumerate(dir_list):
    try:
        with open(i+'/test_metrics.p','r') as f:
            metrics[i] = pickle.load(f)
    except:
        pass      
  
CHILDLESS_VARIABLES = ['V_EP','AUC']
PARENT_VARIABLES = ['Sw','MCC']   
CHILD_VARIABLES = ['thresh','sens','spec','prec','Q2',]
ANALYSIS_VARIABLES = CHILDLESS_VARIABLES + PARENT_VARIABLES + CHILD_VARIABLES
list_analysis = [[m] if m in CHILDLESS_VARIABLES else [m]+CHILD_VARIABLES for M,m in enumerate(CHILDLESS_VARIABLES+PARENT_VARIABLES)]
list_analysis = [m for n in list_analysis for m in n]


dict_init_test = [[[[] for _ in PARENT_VARIABLES] for _ in test_ids] if m in CHILD_VARIABLES else [[] for _ in test_ids] for M,m in enumerate(ANALYSIS_VARIABLES)]
test_save = dict(zip(ANALYSIS_VARIABLES,dict_init_test))

jack.tee('evaluation/text_results.txt','',append=False)
thresh_max_MCC = []
thresh_max_Sw = []
for M,m in enumerate(experiments):
    for I,i in enumerate(name_datasets[:2]):  
        jack.tee('evaluation/text_results.txt','Cmd:\tSaveDir:\tAUC:\tSw:\tthresh:\tsens:\tspec:\tprec:\tQ2:\tMCC:\tMCC:\tthresh:\tsens:\tspec:\tprec:\tQ2:\tSw\t'),
    jack.tee('evaluation/text_results.txt','\n')
    for I,i in enumerate(dir_list):  
        for K,k in enumerate(name_datasets[:2]):  
            tp = np.zeros(thresholds.shape[1])
            fp = np.zeros(thresholds.shape[1])
            fn = np.zeros(thresholds.shape[1])
            tn = np.zeros(thresholds.shape[1])
            for J,j in enumerate(dataset_ids[K]):
                try:
                    threshed_results = np.greater_equal(data[i][m][j]['outputs'],thresholds)
                    labels = data[i][m][j]['masked_labels']
                except:
                    continue
                if threshed_results.shape[0]!=labels.shape[0]:
                    continue
                tp += np.sum(np.logical_and(threshed_results,labels),0)
                tn += np.sum(np.logical_and(np.logical_not(threshed_results),np.logical_not(labels)),0)
                fp += np.sum(np.logical_and(threshed_results,np.logical_not(labels)),0)
                fn += np.sum(np.logical_and(np.logical_not(threshed_results),labels),0) 
            if count > 0:
                sens=jack.sensitivity(tp,tn,fp,fn)
                spec=jack.specificity(tp,tn,fp,fn)
                AUC = np.trapz(sens,spec)
                MCC = jack.MCC(tp,tn,fp,fn)
                Sw = jack.Sw(tp,tn,fp,fn)
                prec = jack.precision(tp,tn,fp,fn)
                #Q2 = jack.accuracy(tp,fn,fp,fn)
                Q2 = np.zeros(prec.shape)
                if k=='Validation':
                    thresh_max_MCC.append(np.nanargmax(MCC))
                    thresh_max_Sw.append(np.nanargmax(Sw))
                Tsw = thresh_max_Sw[M*len(dir_list)+I]
                Tmcc = thresh_max_MCC[M*len(dir_list)+I]
                order = [metrics[i]['cmd'], i, AUC,
                         Tsw*step, Sw[Tsw], sens[Tsw],spec[Tsw],prec[Tsw],Q2[Tsw],MCC[Tsw],
                         Tmcc*step, MCC[Tmcc], sens[Tmcc],spec[Tmcc],prec[Tmcc],Q2[Tmcc],Sw[Tsw]]
                frmt = ['%s','%s','%1.4f',
                        '%1.3f','%1.4f','%1.4f','%1.4f','%1.4f','%1.4f','%1.4f',
                        '%1.3f','%1.4f','%1.4f','%1.4f','%1.4f','%1.4f','%1.4f']
                jack.tee('evaluation/text_results.txt','\t'.join(frmt)%(tuple(order))+'\t'),
    
                
            else:
                print("%s has nothing good for dataset %s"%(i,k))
                bad_data.append([i,k])
        jack.tee('evaluation/text_results.txt','\n')




'''

for J,j in enumerate(experiments):
    print(j)
    for I,i in enumerate(['Validation','Tests']):
        jack.tee('evaluation/text_results.txt','Cmd:\tSaveDir\t'+':\t'.join(list_analysis)+':\t'),
    jack.tee('evaluation/text_results.txt','\n')
    for K,k in enumerate(dir_list):
        for I,i in enumerate(['Validation','Test']):  
            best_epoch = metrics[k][j]['V_EP'][-1][-1]
            formatstr = ['%i' if 'EP' in m else '%1.3f' if 'thresh'==m else '%1.4f' for M,m in enumerate(list_analysis)]
            printstr = [metrics[k][j][m][I][(M-len(CHILDLESS_VARIABLES))/(len(CHILD_VARIABLES)+1)][best_epoch-1] if m in CHILD_VARIABLES else metrics[k][j][m][I][best_epoch-1] for M,m in enumerate(list_analysis)]
            jack.tee('evaluation/text_results.txt','%s\t'%(metrics[k]['cmd'])),
            jack.tee('evaluation/text_results.txt','%s\t'%(k)),
            jack.tee('evaluation/text_results.txt','\t'.join(formatstr)%(tuple(printstr))+'\t'),
        jack.tee('evaluation/text_results.txt','\n')

        
'''
