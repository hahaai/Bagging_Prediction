import os
import sys
import numpy as np
import numpy.ma as ma
import pandas as pd
import time 

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

def standard_bootstrap_orig(dataset,seed=None):
    """
    Generates a bootstrap sample from the input dataset

    Parameters
    ----------
    dataset : array_like
        A matrix of where dimension-0 represents samples

    Returns
    -------
    bdataset : array_like
        A bootstrap sample of the input dataset
    b: list
        list of index for bootstrap

    Examples
    --------
    """
    if not seed:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(seed)

    n = dataset.shape[0]
    b = random_state.randint(0, high=n-1, size=n)
    return dataset[b]


def svr_feature(datax,datay,thres,outfolder,postfix):

    # Random split train and testing 50 times
    for rs in range(1,51):
        print(rs)
        # split train and testing. save the index to apply to others.
        X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(datax, datay,range(0,datax.shape[0]), test_size=0.2, random_state=rs)

        # only select the most significant 5% of features/edges
        X_train,select_index=feature_selection(X_train,y_train,thres)
        X_test=X_test[:,select_index]

        regr = make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1))
        #regr = make_pipeline(SVR(C=1.0, epsilon=0.1))
        regr.fit(X_train, y_train)

        y_predict=regr.predict(X_test)
        np.savetxt(outfolder+'/Test_predict_randomTTS_' + str(rs) + postfix+'.txt',y_predict)
        #print(np.mean(np.abs(y_predict-y_test)))
        #print(np.corrcoef(y_predict,y_test)[0,1])
        np.savetxt(outfolder+'/Test_real_randomTTS_' + str(rs) + postfix+'.txt',y_test)

        # predict on train
        y_predict_train=regr.predict(X_train)
        np.savetxt(outfolder+'/Train_predict_randomTTS_' + str(rs) + postfix+'.txt',y_predict_train)
        #print(np.mean(np.abs(y_predict-y_test)))
        #print(np.corrcoef(y_predict,y_test)[0,1])
        np.savetxt(outfolder+'/Train_real_randomTTS_' + str(rs) + postfix+'.txt',y_train)

def svr_feature_parallel(datax,datay,thres,outfolder,postfix,rs):

    print(rs)
    # split train and testing. save the index to apply to others.
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(datax, datay,range(0,datax.shape[0]), test_size=0.2, random_state=rs)

    # only select the most significant 5% of features/edges
    X_train,select_index=feature_selection(X_train,y_train,thres)
    X_test=X_test[:,select_index]

    regr = make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1))
    #regr = make_pipeline(SVR(C=1.0, epsilon=0.1))
    regr.fit(X_train, y_train)

    y_predict=regr.predict(X_test)
    np.savetxt(outfolder+'/Test_predict_randomTTS_' + str(rs) + postfix+'.txt',y_predict)
    #print(np.mean(np.abs(y_predict-y_test)))
    #print(np.corrcoef(y_predict,y_test)[0,1])
    np.savetxt(outfolder+'/Test_real_randomTTS_' + str(rs) + postfix+'.txt',y_test)

    # predict on train
    y_predict_train=regr.predict(X_train)
    np.savetxt(outfolder+'/Train_predict_randomTTS_' + str(rs) + postfix+'.txt',y_predict_train)
    #print(np.mean(np.abs(y_predict-y_test)))
    #print(np.corrcoef(y_predict,y_test)[0,1])
    np.savetxt(outfolder+'/Train_real_randomTTS_' + str(rs) + postfix+'.txt',y_train)


def feature_selection(x,y,percent):
    corr_base=[]
    for i in range(x.shape[1]):
        #print(i)
        corr_base.append(np.corrcoef(x[:,i],y)[0,1])
        #corr_base.append(ma.corrcoef(ma.masked_invalid(x[:,i]),ma.masked_invalid(y))[0,1])
    corr_base_sort=np.sort(np.abs(corr_base)) 
    selection_idx=np.abs(corr_base)>corr_base_sort[int(round(len(corr_base)*(1-percent)))]
    return x[:,selection_idx],selection_idx

def ts_bootstrap_svr(data,pheno,ts_seed,datapath):
    # the data will be for subidx*(time point*parcel), parcel is always 998

    dim1=data.shape[1]/998
    dim2=998

    dataall_bagging=np.zeros(shape=(data.shape[0],998*997/2))
    for subidx in range(0,data.shape[0]):
        #print(subidx)
        subdata=data[subidx,:]
        subdata=subdata.reshape((dim1,dim2))
        block_size=np.int(np.floor(np.sqrt(subdata.shape[0]))) # for time series bootstrap
        if ts_seed == 0:
            data_corr=np.corrcoef(np.transpose(subdata))
            data_corr=upper_tri_indexing(data_corr)
            dataall_bagging[subidx,:]=data_corr
        else:
            dataall_bagging[subidx,:]=timeseries_bootstrap(subdata, block_size, ts_seed)
       
    outfolder=datapath+'TimeSeriesBS-'+str(ts_seed)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    svr_feature(dataall_bagging,pheno['Age_in_Yrs'],0.05,outfolder,'')


def Subject_Time_series_Bootstrapping_aggerate(data,subidx):
    dim1=data.shape[1]/998
    dim2=998
    TS_BS = True
    #dataall_bagging=np.zeros(shape=(data.shape[0],998*997/2))
    print(subidx)
    subdata=data[subidx,:]
    subdata=subdata.reshape((dim1,dim2))
    block_size=np.int(np.floor(np.sqrt(subdata.shape[0]))) # for time series bootstrap

    if TS_BS == True:
        for ts_seed in range(1,(num_tsb+1)):
            if ts_seed == 1:
                tmp=timeseries_bootstrap(subdata, block_size, ts_seed)
            else:
                tmp=tmp + timeseries_bootstrap(subdata, block_size, ts_seed)
        tmp = tmp/float(num_tsb)

    if TS_BS == False:
        data_corr=np.corrcoef(np.transpose(subdata))
        tmp=upper_tri_indexing(data_corr)
    return tmp

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

group_bagging=True
ts_bagging=True

Aggregate_type='FC'

num_tsb=100
num_grb=0
num_jobs=50

outpath='/home/ec2-user/Results_TS_BS_only/'

dataprefix='/home/ec2-user/saveddata'

for timeduration in [60,30,10,5]:
#for timeduration in [10]:

    # load the pre-saved data and sublist
    # /data3/aki/bagging_prediction/saveddata/time_duraiton_5.npy
    print('Loadign the data')
    #data_orig=load(dataprefix+'/memmap'+str(timeduration),mmap_mode='r')

    data_filename_memmap = '/home/ec2-user/saveddata/memmap' + str(timeduration)
    data = load(data_filename_memmap, mmap_mode='r')


    suball=np.load('/home/ec2-user/saveddata/final_sublist_new.npy')
    #data_orig=data_orig[0:len(suball),:]
    print('data loading done')
    if group_bagging == True and ts_bagging==True:
        for gb in range(0,1):# group/subjects bagging 100 times, this gb is the seed for bootstrap
            print('group bagging:' + str(gb))
            if gb ==0:
                #data=data_orig
                sublist=suball
            else:
                sublist=standard_bootstrap_orig(suball, gb)
                data=standard_bootstrap_orig(data_orig, gb)

            sublist_df=pd.DataFrame(sublist,columns=['Subject'])
            pheno=sublist_df.merge(pheno_orig,on='Subject',how='left')
            
            outfolder=outpath +'/' + 'Aggregate_'+Aggregate_type +'/time_duraiton_'+str(timeduration)+'/GroupBS_'+str(gb)+'/'

            #outfolder=outpath + '/time_duraiton_'+str(timeduration)+'_GroupBS_'+str(gb)+'_Aggregate_'+Aggregate_type
            if not os.path.isdir(outfolder):
                 os.makedirs(outfolder)

            print('Start doing model')
            tsb=1
            print('start bootstraap_svr')
            #ts_bootstrap_svr(data,pheno,tsb,outfolder)
            





            if Aggregate_type == 'Prediction':
                #Parallel(n_jobs=num_jobs,backend="threading")(delayed(ts_bootstrap_svr)(data,pheno,tsb,outfolder) for tsb in range(1,(num_tsb+1)))
                #Parallel(n_jobs=num_jobs)(delayed(ts_bootstrap_svr)(data,pheno,tsb,outfolder) for tsb in range(0,num_tsb+1))
                Parallel(n_jobs=num_jobs)(delayed(ts_bootstrap_svr)(data,pheno,tsb,outfolder) for tsb in range(281,461))


            
            # do SVR on aggregated correlation matrix,
            if Aggregate_type =='FC':


                dataall_bagging_tmp = Parallel(n_jobs=num_jobs)(delayed(Subject_Time_series_Bootstrapping_aggerate)(data,subidx) for subidx in range(0,data.shape[0]))  
                dataall_bagging=np.zeros(shape=(data.shape[0],998*997/2))
                for ii in range(0,dataall_bagging.shape[0]):
                     dataall_bagging[ii,:]=dataall_bagging_tmp[ii]

                np.save('Corr_matrix/Correlation_matrix_Base_'+str(timeduration)+'_min.npy',dataall_bagging)

                outfolder=outpath +'/' + 'Aggregate_'+Aggregate_type +'/time_duraiton_'+str(timeduration)+'/GroupBS_'+str(gb)
                if not os.path.isdir(outfolder):
                    os.makedirs(outfolder)

                Parallel(n_jobs=num_jobs)(delayed(svr_feature_parallel)(dataall_bagging,pheno['Age_in_Yrs'],0.05,outfolder,'',tts) for tts in range(1,51))

