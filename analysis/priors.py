import deepdish as dd
import constants as C
def find_verb(file_name):
    v = ''
    for c in file_name:
        if (v + c).isalpha():
            v = v + c
        else:
            break

    return v

def per_verb_performance():
    per_verb_train = {}
    phase = 'train'

    IQA_dataset = dd.io.load('{0}vqa_{1}_info.h5'.format(C.data_dir, phase))

    for i in range(len(IQA_dataset['I'])):
        v = find_verb(IQA_dataset['I'][i])
        a = IQA_dataset['A'][i]
        if v not in per_verb_train.keys():
            per_verb_train[v] = {}

        if a not in per_verb_train[v].keys():
            per_verb_train[v][a] = 0
        per_verb_train[v][a] = per_verb_train[v][a] + 1

    top_v_train = {}
    for v in per_verb_train.keys():
        top_v_train[v] = sorted(per_verb_train[v].iteritems(), key=lambda (k, v): (v, k), reverse=True)[0]

    per_verb_test = {}
    phase = 'test'

    IQA_dataset = dd.io.load('{0}vqa_{1}_info.h5'.format(C.data_dir, phase))
    prior = 0
    prior_verb = 0
    counter_prior_verb = {}
    prior_per_verb = {}

    ansewrs = []

    for i in range(len(IQA_dataset['I'])):
        v = find_verb(IQA_dataset['I'][i])
        a = IQA_dataset['A'][i]

        # if v not in per_verb_test.keys():
        #     per_verb_test[v] = {}

        # if a not in per_verb_test[v].keys():
        #     per_verb_test[v][a] = 0
        # per_verb_test[v][a] = per_verb_test[v][a] + 1

        if a == 'outdoors':
            prior = prior + 1

        if v not in prior_per_verb.keys():
            prior_per_verb[v] = 0
            counter_prior_verb[v] = 0

        counter_prior_verb[v] +=1

        if a == top_v_train[v][0]:
            prior_per_verb[v] = prior_per_verb[v] + 1
            prior_verb = prior_verb + 1

    for i in range(len(IQA_dataset['I'])):
        v = find_verb(IQA_dataset['I'][i])
        ansewrs.append(top_v_train[v][0])

    # top_v_test = {}
    # for v in top_v_test.keys():
    #     top_v_test[v] = sorted(top_v_test[v].iteritems(), key=lambda (k, v): (v, k), reverse=True)[0]

    print top_v_train
    print prior, len(IQA_dataset['I'])
    print prior_verb, len(IQA_dataset['I'])

    f = open(C.data_dir + C.priors + C.answers_file_name, 'w')
    f.write('\n'.join(ansewrs))
    f.close()

    f = open(C.data_dir + C.priors + 'verb_priors.txt', 'w')
    records = []
    for v in counter_prior_verb:
        records.append('{0},{1},{2},{3}'.format(v, top_v_train[v][0],top_v_train[v][1], counter_prior_verb[v]))
    f.write('\n'.join(records))
    f.close()

def get_test_verbs():
    phase = 'test'

    IQA_dataset = dd.io.load('{0}vqa_{1}_info.h5'.format(C.data_dir, phase))
    verbs = []
    for i in range(len(IQA_dataset['I'])):
        v = find_verb(IQA_dataset['I'][i])
        verbs.append(v)

    f = open(C.data_dir + C.priors + 'test_verbs.txt', 'w')
    f.write('\n'.join(verbs))
    f.close()

phase = 'test'

IQA_dataset = dd.io.load('{0}vqa_{1}_info.h5'.format(C.data_dir, phase))
roles = []
for i in range(len(IQA_dataset['AR'])):
    r = IQA_dataset['AR'][i]
    roles.append(r)

f = open(C.data_dir + C.priors + 'test_roles.txt', 'w')
f.write('\n'.join(roles))
f.close()
