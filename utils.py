import numpy as np
import torch
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.util import ngrams
from math import log
import copy


# if test=True, load the test file, or load the whole file
def load_data(Dataset, test=True):
    if test:
        test_idx = pickle.load(open(Test_Idx_File[Dataset], 'rb'))
        whole_data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
        whole_label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
        data = whole_data[test_idx]
        label = whole_label[test_idx]
        return data, label
    data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
    label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
    return data, label

def dataset_split(Dataset):
    train_idx = pickle.load(open('./dataset/'+Dataset+'_train_idx.pickle', 'rb'))
    test_idx = pickle.load(open('./dataset/'+Dataset+'_test_idx.pickle', 'rb'))
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=666)
    return train_idx, val_idx, test_idx


# After attack, summarize the success rate, changed num and so on
def write_file(Dataset, Model_Type, budget, algorithm, algo_name, time_limit):
    log_f = open('./Logs/%s/MF_%s_%d_%s.bak' % (Dataset, Model_Type, budget, algorithm), 'w+')
    TITLE = '=== ' + Dataset + Model_Type + str(budget) + algorithm + ' time = ' + str(time_limit) + ' ==='
    print(TITLE, file=log_f, flush=True)

    directory = './Logs/%s/%s/%s' % (Dataset, Model_Type, algo_name)
    print()
    print(directory)
    print(directory, file=log_f, flush=True)
    mf_process_temp = pickle.load(open(directory + 'mf_process_%d.pickle' % budget, 'rb'))
    changed_set_process_temp = pickle.load(open(directory + 'changed_set_process_%d.pickle' % budget, 'rb'))
    robust_flag = pickle.load(open(directory + 'robust_flag_%d.pickle' % budget, 'rb'))
    query_num = pickle.load(open(directory + 'querynum_%d.pickle' % budget, 'rb'))
    time = pickle.load(open(directory + 'time_%d.pickle' % budget, 'rb'))
    iteration_file = pickle.load(open(directory + 'iteration_%d.pickle' % budget, 'rb'))
    funccall_all = pickle.load(open(directory + 'modified_funccall_%d.pickle' % budget, 'rb'))
    mf_process = []
    changed_set_process = []
    time_attack = []
    query_num_attack = []
    flip_changed_num = []
    iteration = []
    robust = 0
    for j in range(len(robust_flag)):
        if robust_flag[j] == 0:
            mf_process.append(mf_process_temp[j])
            changed_set_process.append(changed_set_process_temp[j])
            time_attack.append(time[j])
            query_num_attack.append(query_num[j])
            flip_changed_num.append(len(changed_set_process_temp[j][-1]))
            iteration.append(iteration_file[j])
        elif robust_flag[j] == 1:
            robust += 1

    sorted_flip_changed_num = np.sort(flip_changed_num)
    if sorted_flip_changed_num == np.array([]):
        change_medium = 0
    else:
        change_medium = sorted_flip_changed_num[len(flip_changed_num) // 2]

    print('success rate:', len(iteration) / len(mf_process_temp))
    print('average iteration:', np.mean(iteration))
    print('average changed code', np.mean(flip_changed_num))
    print('average time:', np.mean(time_attack))
    print('average query number', np.mean(query_num_attack))
    print('medium changed number', change_medium)
    print('clean test data accuracy:', len(robust_flag) / len(funccall_all))
    print('adversarial accuracy:', robust / len(funccall_all))

    print('success rate:', len(iteration) / len(mf_process_temp), file=log_f, flush=True)
    print('average iteration:', np.mean(iteration), file=log_f, flush=True)
    print('average changed code', np.mean(flip_changed_num), file=log_f, flush=True)
    print('average time:', np.mean(time_attack), file=log_f, flush=True)
    print('average query number', np.mean(query_num_attack), file=log_f, flush=True)
    print('medium changed number', change_medium, file=log_f, flush=True)
    print('clean test data accuracy:', len(robust_flag) / len(funccall_all), file=log_f, flush=True)
    print('adversarial accuracy:', robust / len(funccall_all), file=log_f, flush=True)
    print('end')


def get_decoder_pars(model):
  pars = []
  par_names = []
  for n, m in model.named_parameters():
    if 'encoder' in n:
      continue
    pars.append(m)
    par_names.append(n)

  return pars, par_names


# make sure the path exist
def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)


# make some vector one hot vector
def one_hot_labels(y, n_labels):
    return torch.zeros(y.size(0), n_labels).long().scatter(1, y.unsqueeze(1).cpu(), 1).cuda()


def one_hot_samples(x, dataset):
    return torch.zeros(x.size(0), x.size(1), num_category[dataset]).long().scatter(2, x.unsqueeze(2).cpu(), 1)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def input_process(batch_diagnosis_codes, Dataset):
    if Dataset_type[Dataset] == 'multi':
        t_diagnosis_codes = one_hot_samples(batch_diagnosis_codes, Dataset).cuda().float()
    elif Dataset_type[Dataset] == 'binary':
        t_diagnosis_codes = batch_diagnosis_codes.cuda().float()
    else:
        t_diagnosis_codes = batch_diagnosis_codes.cuda().float()
    return t_diagnosis_codes


Test_Idx_File = {
    'Splice': './dataset/Splice_test_idx.pickle',
    'pedec': './dataset/pedec_test_idx.pickle',
    'census': './dataset/census_test_attack_idx.pickle',
}

Train_Idx_File = {
    'Splice': './dataset/Splice_train_idx.pickle',
    'pedec': './dataset/pedec_train_idx.pickle',
    'census': './dataset/census_train_idx.pickle',    
}

Whole_Data_File = {
    'Splice': './dataset/SpliceX.pickle',
    'pedec': './dataset/pedecX.pickle',
    'census': './dataset/censusX.pickle',
}

Whole_Label_File = {
    'Splice': './dataset/SpliceY.pickle',
    'pedec': './dataset/pedecY.pickle',
    'census': './dataset/censusY.pickle',
}

num_category = {'Splice': 5, 'pedec': 3, 'census': 52}
num_feature = {'Splice': 60, 'pedec': 5000, 'census': 41}
num_samples = {'Splice': 3190, 'pedec': 21790, 'census': 299285}
num_avail_category = {'Splice': 4, 'pedec': 2, 'census': 51}
num_classes = {'Splice': 3, 'pedec': 2, 'census': 2}

# the model parameters for each model
Splice_Model = {
    'MLP': './classifier/Splice_MLP.par',
    'MLP_IGR': './classifier/Splice_MLP_IGR.par',
    'MLP_SG': './classifier/Splice_MLP_SG.par',
    'MLP_IG': './classifier/Splice_MLP_IG.par',
    'MLP_IGSG_VG': './classifier/Splice_MLP_IGSG_VG.par',
    'MLP_IGSG_VSG': './classifier/Splice_MLP_IGSG_VSG.par',
    'MLP_Adv': './classifier/Splice_MLP_Adv.par',
    'MLP_AFD': './classifier/Splice_MLP_AFD.par',
    'MLP_FastBAT': './classifier/Splice_MLP_FastBAT.par',
    'MLP_SGSG': './classifier/Splice_MLP_SGSG.par',
    'MLP_IGIG': './classifier/Splice_MLP_IGIG_m1.par',
    'MLP_JR': './classifier/Splice_MLP_JR.par',
    'MLP_IGSG': './classifier/Splice_MLP_IGSG.par',
}

pedec_Model = {
    'MLP': './classifier/pedec_MLP.par',
    'MLP_Adv': './classifier/pedec_MLP_Adv.par',
    'MLP_IGR': './classifier/pedec_MLP_IGR.par',
    'MLP_SG': './classifier/pedec_MLP_SG.par',
    'MLP_IG': './classifier/pedec_MLP_IG.par',
    'MLP_IGSG_VG': './classifier/pedec_MLP_IGSG_VG.par',
    'MLP_IGSG_VSG': './classifier/pedec_MLP_IGSG_VSG.par',
    'MLP_IGSG': './classifier/pedec_MLP_IGSG.par',
    'MLP_AFD': './classifier/pedec_MLP_AFD.par',
    'MLP_FastBAT': './classifier/pedec_MLP_FastBAT.par',
    'MLP_SGSG': './classifier/pedec_MLP_SGSG.par',
    'MLP_IGIG': './classifier/pedec_MLP_IGIG_m1.par',
    'MLP_JR': './classifier/pedec_MLP_JR.par',
}

census_Model = {
    'MLP': './classifier/census_MLP.par',
    'MLP_Adv': './classifier/census_MLP_Adv.par',
    'MLP_IGR': './classifier/census_MLP_IGR.par',
    'MLP_SG': './classifier/census_MLP_SG.par',
    'MLP_IG': './classifier/census_MLP_IG.par',
    'MLP_IGSG_VG': './classifier/census_MLP_IGSG_VG.par',
    'MLP_IGSG_VSG': './classifier/census_MLP_IGSG_VSG.par',
    'MLP_AFD': './classifier/census_MLP_AFD.par',
    'MLP_FastBAT': './classifier/census_MLP_FastBAT.par',
    'MLP_SGSG': './classifier/census_MLP_SGSG.par',
    'MLP_IGIG': './classifier/census_MLP_IGIG_m1.par',
    'MLP_JR': './classifier/census_MLP_JR.par',
    'MLP_IGSG': './classifier/census_MLP_IGSG.par',
}

census_category = [5, 9, 4, 4, 17, 4, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 4, 4, 4, 6, 6,
                   51, 38, 8, 3, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 4, 2]

complex_categories = {
    'census': census_category
}

Model = {
    'Splice': Splice_Model,
    'pedec': pedec_Model,
    'census': census_Model
}

Dataset_type = {'Splice': 'multi',
                'pedec': 'multi',
                'census': 'multi',
                }

budgets = {'Splice': 5, 'pedec': 6, 'census': 3}
OMPGS_time_limits = {'Splice': 1, 'pedec': 5, 'census': 0.1}
FSGS_time_limits = {'Splice': 10, 'pedec': 150, 'census': 10}


# return the model parameter
def model_file(Dataset, Model_Type):
    return Model[Dataset][Model_Type]
