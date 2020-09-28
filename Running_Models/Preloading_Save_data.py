import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
from joblib import dump, load

def upper_tri_indexing(A):
    m = A.shape[0]

    r,c = np.triu_indices(m,1)
    return A[r,c]

def timeseries_bootstrap(tseries, block_size, seed=None):
    """
    Generates a bootstrap sample derived from the input time-series.
    Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks
    random_state : integer
        the random state to seed the bootstrap

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries


    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
       changes in brain connectivity for functional MRI. Statistica Sinica,
       special issue on Statistical Challenges and Advances in Brain Science,
       2008, 18: 1253-1268.

    Examples
    --------

    >>> x = np.arange(50).reshape((5, 10)).T
    >>> sample_bootstrap(x, 3)
    array([[ 7, 17, 27, 37, 47 ],
           [ 8, 18, 28, 38, 48 ],
           [ 9, 19, 29, 39, 49 ],
           [ 4, 14, 24, 34, 44 ],
           [ 5, 15, 25, 35, 45 ],
           [ 6, 16, 26, 36, 46 ],
           [ 0, 10, 20, 30, 40 ],
           [ 1, 11, 21, 31, 41 ],
           [ 2, 12, 22, 32, 42 ],
           [ 4, 14, 24, 34, 44 ]])

    """
    import numpy as np

    if not seed:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(seed)

    # calculate number of blocks
    k = int(np.ceil(float(tseries.shape[0]) / block_size))

    # generate random indices of blocks
    r_ind = np.floor(random_state.rand(1, k) * tseries.shape[0])
    blocks = np.dot(np.arange(0, block_size)[:, np.newaxis], np.ones([1, k]))

    block_offsets = np.dot(np.ones([block_size, 1]), r_ind)
    block_mask = (blocks + block_offsets).flatten('F')[:tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])

    #return tseries[block_mask.astype('int'), :], block_mask.astype('int')
    data=tseries[block_mask.astype('int'), :]
    data_corr=np.corrcoef(np.transpose(data))
    data_corr=upper_tri_indexing(data_corr)
    return data_corr

def standard_bootstrap(sublist, seed=None):
    """
    Generates a bootstrap sample from the input dataset

    Parameters
    ----------
    dataset : list_like
        A matrix of where dimension-0 represents samples
    random_state : integer
        the random state to seed the bootstrap

    Returns
    -------
    bdataset : array_like
        A bootstrap sample of the input dataset

    Examples
    --------
    """

    if not seed:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(seed)

    n = len(sublist)
    b = random_state.randint(0, high=n - 1, size=n)
    
    return [sublist[i] for i in b]


def svr_feature(datax,datay,thres,outfolder,postfix):
    # only select the most significant 5% of features/edges
    datax=feature_selection(datax,datay,thres)
    # Random split train and testing 50 times
    for rs in range(1,51):
        print(rs)
        # split train and testing. save the index to apply to others.
        X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(datax, datay,range(0,datax.shape[0]), test_size=0.2, random_state=rs)
        regr = make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1))
        #regr = make_pipeline(SVR(C=1.0, epsilon=0.1))
        regr.fit(X_train, y_train)

        y_predict=regr.predict(X_test)
        np.savetxt(outfolder+'/predict_randomTTS_' + str(rs) + postfix+'.txt',y_predict)
        #print(np.mean(np.abs(y_predict-y_test)))
        #print(np.corrcoef(y_predict,y_test)[0,1])
        np.savetxt(outfolder+'/real_randomTTS_' + str(rs) + postfix+'.txt',y_predict)

def feature_selection(x,y,percent):
    corr_base=[]
    for i in range(x.shape[1]):
        corr_base.append(np.corrcoef(x[:,i],y)[0,1])
    corr_base_sort=np.sort(np.abs(corr_base)) 
    return x[:,np.abs(corr_base)>corr_base_sort[round(len(corr_base)*(1-percent))]]



######### start
datafolder='/home/ec2-user/saveddata/indiv_data'

## run though subjects once to get the number of subjects in total
suball=[]
for sub in os.listdir(datafolder):
    sub_status=1
    for acq in ['REST1_LR','REST2_LR','REST1_RL','REST2_RL']:
        file_base=datafolder +'/' + sub + '/rfMRI_'+acq+'_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries_TS_Parcel.txt'
        if not os.path.isfile(file_base):
            sub_status=0
    if sub_status==1:
        suball.append(sub)
    else:
        print(sub)


## for testing, limit the subject number
#suball=suball[1:4]

# load in pheno
pheno_file='/home/ec2-user/saveddata/HCP_1_Openaccess_demoinfo.csv'
pheno_orig=pd.read_csv(pheno_file)
pheno_orig=pheno_orig[['Subject','Age_in_Yrs']]
pheno_orig['Subject']=pheno_orig['Subject'].apply(str)


### analysis folking:
# Time duration: 5, 10, 30, 60, number of volumne: 400,800, 1190*2, 1190*4
# grouping bagging
# time series bagging
# random splitting of training-testing dataset

group_bagging=False
ts_bagging=False
num_tsb=2
num_grb=100
num_jobs=50


sublist=suball
suball_2save=suball
sublist_df=pd.DataFrame(sublist,columns=['Subject'])
pheno=sublist_df.merge(pheno_orig,on='Subject',how='left')


## initialize

sublist=sublist[0:5]

data2save_5=np.zeros(shape=(len(sublist),400*998)) # when reload the data, need to reshape it to (400,998)
data2save_10=np.zeros(shape=(len(sublist),800*998))
data2save_30=np.zeros(shape=(len(sublist),2380*998))
data2save_60=np.zeros(shape=(len(sublist),4760*998))

idx=0
subnew=[]
for sub in sublist:
    print(str(idx) + ' - '+ sub)
    file1=datafolder +'/' + sub + '/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries_TS_Parcel.txt'
    file2=datafolder +'/' + sub + '/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries_TS_Parcel.txt'
    file3=datafolder +'/' + sub + '/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries_TS_Parcel.txt'
    file4=datafolder +'/' + sub + '/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries_TS_Parcel.txt'
    data1=np.loadtxt(file1,delimiter=',')
    data2=np.loadtxt(file2,delimiter=',')
    data3=np.loadtxt(file3,delimiter=',')
    data4=np.loadtxt(file4,delimiter=',')
    
    if data1.shape[0] != 1190 or data2.shape[0] != 1190 or data3.shape[0] != 1190 or data4.shape[0] != 1190:
    #if data1.shape[0] != 1190:

        suball_2save.remove(sub)
        #idx += 1
        continue

    for timeduration in [5,10,30,60]:
    #for timeduration in [5]:

        if timeduration == 5:
            data=data1[0:400,:]
            data2save_5[idx,:]=data.flatten()

        if timeduration == 10:
            data=data1[0:800,:]
            data2save_10[idx,:]=data.flatten()

        if timeduration == 30:
            data=np.concatenate([data1,data2])
            data2save_30[idx,:]=data.flatten()

        if timeduration == 60:
            data=np.concatenate([data1,data2,data3,data4])
            data2save_60[idx,:]=data.flatten()
    subnew.append(sub)
    idx += 1

   



####
data_filename_memmap = '/home/ec2-user/saveddata/memmap5'
dump(data2save_5, data_filename_memmap)
#data = load(data_filename_memmap, mmap_mode='r')

data_filename_memmap = '/home/ec2-user/saveddata/memmap10'
dump(data2save_10, data_filename_memmap)

data_filename_memmap = '/home/ec2-user/saveddata/memmap30'
dump(data2save_30, data_filename_memmap)

####
data_filename_memmap = '/home/ec2-user/saveddata/memmap60'
dump(data2save_60, data_filename_memmap)
#data = load(data_filename_memmap, mmap_mode='r')



# save sublist
outfile='/home/ec2-user/saveddata/final_sublist_new.npy'
np.save(outfile,subnew)



