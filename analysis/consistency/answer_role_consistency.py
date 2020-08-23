import numpy as np
import json
from collections import Counter
# import h5py
import deepdish as dd
import constants as C
import tensorflow as tf
from tensorflow.python.keras import backend as K_backend


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 8.0
K_backend.set_session(tf.Session(config=config))

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_metadata():
    meta_data = json.load(open(C.data_dir + C.vocabulary_file, 'r'))
    meta_data['ix_to_word'] = {word: int(i) for i, word in meta_data['ix_to_word'].items()}
    meta_data['ix_to_role'] = {role: int(i) for i, role in meta_data['ix_to_role'].items()}
    return meta_data

def get_answer_role_freq():
    with open(C.data_dir + C.answer_role_freq_file) as f:
        answer_role_freq = json.load(f)

    return answer_role_freq

def compute_answer_role_freq(meta, type='test'):
    print("Reading Data...")

    IQA_dataset = dd.io.load(C.data_dir + 'vqa_{0}_info.h5'.format(type))
    img_data = IQA_dataset['X'][0]
    ques_data = IQA_dataset['X'][1]
    ans_data = IQA_dataset['Y']
    role_data = IQA_dataset['Yrole']

    ans_counter = {}
    ans_counter['OTHER']=[]
    for k in meta['ans_to_ix']:
        ans_counter[k]=[]

    for i in range(len(ans_data)):
        # print ' . . . '
        ans_i = np.argmax(ans_data[i])
        rol_i = np.argmax(role_data[i])

        if np.max(ans_data[i])==0:
            # ans_counter['OTHER'].append(meta['ix_to_role'][rol_i])
            continue

        ans_counter[meta['ix_to_ans'][str(ans_i)]].append(meta['ix_to_role'][rol_i])

    # sk = [ (i,j) for a, l in sorted([(x,len(y)) for x, y in ans_counter.iteritems()] , reverse=True, key=lambda z: z[1])]

    answer_role_freq = {}
    for k in meta['ans_to_ix']:
        n = len(ans_counter[k])
        ctr = Counter(ans_counter[k])
        answer_role_freq[k]=ctr
    #
    #     print k, n , len(ans_data)
    #     print ctr
    #     print '   '
    #
    # print 'OTHER', len(ans_counter['OTHER'])
    # print Counter(ans_counter['OTHER'])
    # print '  '

    with open(C.data_dir + C.answer_role_freq_file, 'w') as f:
        json.dump(answer_role_freq, f)

    # print len(ans_data)

def read_answers():
    answers=[]
    with open(C.data_dir + C.experiment_baseline + C.wups + C.answers_file_name, 'r') as f0:
        lines = f0.readlines()
        for l in lines:
            answers.append(l.replace('\n', ''))

    return answers

def read_roles():
    roles=[]
    with open(C.data_dir + C.experiment_baseline + C.wups + C.roles_file_name, 'r') as fr:
        lines = fr.readlines()
        for l in lines:
            roles.append(l.replace('\n', ''))

    return roles

def read_predictions(mode):
    predictions=[]
    if mode==C.experiment_baseline:
        with open(C.data_dir+C.experiment_baseline+ C.wups + C.predictions_file_name, 'r') as f1:
            lines = f1.readlines()
            for l in lines:
                predictions.append(l.replace('\n',''))
    elif mode==C.experiment_multi:
        with open(C.data_dir+C.experiment_multi+ C.wups + C.predictions_file_name, 'r') as f2:
            lines = f2.readlines()
            for l in lines:
                predictions.append(l.replace('\n',''))
    else:
        return []

    return predictions

def compute_role_answers(answer_role_freq):
    role_answers={}
    for ans in answer_role_freq:
        roles=answer_role_freq[ans].keys()
        for r in roles:
            if r not in role_answers.keys():
                role_answers[r]=set([])
            role_answers[r].add(ans)

    return role_answers

def compute_role_answers_top5(answer_role_freq):
    role_answers={}

    for ans in answer_role_freq:
        roles=answer_role_freq[ans].keys()
        t = min(5,len(roles))
        top5_roles = [k for (k,v) in sorted(answer_role_freq[ans].items(), reverse=True, key=lambda kv: (kv[1], kv[0]))][:t]
        print top5_roles
        for r in top5_roles:
            if r not in role_answers.keys():
                role_answers[r]=set([])
            role_answers[r].add(ans)

    return role_answers

def compute_consistency():
    answer_role_freq = get_answer_role_freq()
    role_answers= compute_role_answers(answer_role_freq)
    # role_answers= compute_role_answers_top5(answer_role_freq)

    answers= read_answers()
    roles=read_roles()

    cnn_lstm_predictions = read_predictions(C.experiment_baseline)
    clp_acc=0
    clp_role_acc=0
    multi_cnn_lstm_predictions = read_predictions(C.experiment_multi)
    mult_clp_acc=0
    mult_role_clp_acc=0

    print len(answers)
    print len(cnn_lstm_predictions)
    print len(multi_cnn_lstm_predictions)
    if len(cnn_lstm_predictions)!=len(multi_cnn_lstm_predictions) or len(answers) !=len(multi_cnn_lstm_predictions) or len(answers)!=len(roles):
        raise Exception()

    for i in range(len(answers)):
        if answers[i]==cnn_lstm_predictions[i]:
            clp_acc+=1
            clp_role_acc+=1
        else:
            if roles[i] in role_answers:
                if cnn_lstm_predictions[i] in role_answers[roles[i]]:
                    clp_role_acc+=1

        if answers[i] == multi_cnn_lstm_predictions[i]:
            mult_clp_acc+=1
            mult_role_clp_acc+=1
        else:
            if roles[i] in role_answers:
                if multi_cnn_lstm_predictions[i] in role_answers[roles[i]]:
                    mult_role_clp_acc+=1

    print clp_acc,'/',len(answers),' = ',float(clp_acc)/len(answers)
    print clp_role_acc,'/',len(answers),' = ',float(clp_role_acc)/len(answers)

    print mult_clp_acc,'/',len(answers),' = ',float(mult_clp_acc)/len(answers)
    print mult_role_clp_acc,'/',len(answers),' = ',float(mult_role_clp_acc)/len(answers)



def run_code(type='train'):
    # meta = get_metadata()
    # meta['ix_to_role'] = {y: x for x, y in meta['ix_to_role'].iteritems()}
    # compute_answer_role_freq(meta,type)

    answer_role_freq = get_answer_role_freq()

    sorted_freq= sorted([(k,len(v)) for k,v in answer_role_freq.items()], key=lambda x:x[1], reverse=True)
    print len(sorted_freq)
    for (a,f) in sorted_freq:
        print a,f
        print answer_role_freq[a]

    # for a in answer_role_freq:
    #     print a
    #     print answer_role_freq[a]

    # compute_consistency()
