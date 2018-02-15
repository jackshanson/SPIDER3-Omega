import os,sys,glob

dir_list = glob.glob('./save_files_large/*')
id = [int(i[-2:].strip('_')) for i in dir_list]
id,dir_list = zip(*sorted(zip(id, dir_list)))


for I,i in enumerate(dir_list):    
    try:
        f = open(i+'/results_log.txt','r')
        a = f.read().splitlines()
        cmd = a[-5]
        all_results = a[-3]
        pro_results = a[-1]
        print('%s\t%s\t%s\t%s'%(cmd,i,all_results,pro_results))
        f.close()
    except:
        pass
