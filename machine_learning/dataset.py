from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
import numpy as np
import scipy.io as sio


def get_data_deap_V(sample_path, index, i):
    nums=np.zeros(32)
    target_sample = np.load(sample_path + 'person_%d data.npy' % i)
    target_elabel = np.load(sample_path + 'person_%d label_V.npy' % i)
    nums[i] = target_sample.shape[0]
    source_sample = []
    source_elabel = []
    print("train:", index)
    for j in index:
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'person_%d label_V.npy' % j)
        nums[j]=t_source_sample.shape[0]
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    l1,l2=target_elabel.shape[0],source_elabel.shape[0]
    print("target 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(target_elabel == 3)/l1,
          np.sum(target_elabel == 2)/l1,
          np.sum(target_elabel == 1)/l1,
          np.sum(target_elabel == 0)/l1))
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    print("sample nums:",nums)
    return source_sample.reshape([source_sample.shape[0],-1]),source_elabel,target_sample.reshape([target_sample.shape[0],-1]),target_elabel

def get_data_deap_A(sample_path, index, i):
    nums=np.zeros(32)
    target_sample = np.load(sample_path + 'person_%d data.npy' % i)
    target_elabel = np.load(sample_path + 'person_%d label_A.npy' % i)
    nums[i] = target_sample.shape[0]
    source_sample = []
    source_elabel = []
    print("train:", index)
    for j in index:
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'person_%d label_A.npy' % j)
        nums[j]=t_source_sample.shape[0]
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    l1,l2=target_elabel.shape[0],source_elabel.shape[0]
    print("target 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(target_elabel == 3)/l1,
          np.sum(target_elabel == 2)/l1,
          np.sum(target_elabel == 1)/l1,
          np.sum(target_elabel == 0)/l1))
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    print("sample nums:",nums)
    return source_sample.reshape([source_sample.shape[0],-1]),source_elabel,target_sample.reshape([target_sample.shape[0],-1]),target_elabel

def get_data_seed(sample_path, index, i):
    nums=np.zeros(15)
    target_sample = np.load(sample_path + 'person_%d data.npy' % i)
    target_elabel = np.load(sample_path + 'label.npy')
    nums[i] = target_sample.shape[0]
    source_sample = []
    source_elabel = []
    print("train:", index)
    for j in index:
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'label.npy')
        nums[j]=t_source_sample.shape[0]
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    l1,l2=target_elabel.shape[0],source_elabel.shape[0]
    print("target 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(target_elabel == 3)/l1,
          np.sum(target_elabel == 2)/l1,
          np.sum(target_elabel == 1)/l1,
          np.sum(target_elabel == 0)/l1))
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    print("sample nums:",nums)
    return source_sample.reshape([source_sample.shape[0],-1]),source_elabel,target_sample.reshape([target_sample.shape[0],-1]),target_elabel


def get_data(data_type="SEED",i=0):
    SEED_PATH="/data/EEG/DE_4D/"
    DEAP_PATH="/data/EEG/Channel_DE_sample/Binary/"

    if data_type=="SEED":
        index=[j for j in range(15)]
        del index[i]
        return get_data_seed(SEED_PATH,index,i)
    elif data_type=="DEAP_A":
        index=[j for j in range(32)]
        del index[i]
        return get_data_deap_A(DEAP_PATH,index,i)
    elif data_type=="DEAP_V":
        index=[j for j in range(32)]
        del index[i]
        return get_data_deap_V(DEAP_PATH,index,i)