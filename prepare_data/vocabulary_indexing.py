import json
import constants as C
import os.path
import qa_temp_realization as qatr

def create_vocabulary(set_of_all_roles, top_1000_ans, phase='train'):
    dataset = qatr.realize_imSitu_abstract_question_realization()[phase]

    if os.path.isfile('{0}{1}'.format(C.data_dir, C.vocabulary_file)):
        return load_vocabulary()

    Vocabulary = set([])
    V_dataset = {}
    if phase == 'train':
        V_dataset['ix_to_role']={}
        V_dataset['role_to_ix']={}
        index = 0
        # V_dataset['ix_to_role'][index] = ''
        # V_dataset['role_to_ix'][''] = index
        # index = index + 1
        for r in set_of_all_roles:
            V_dataset['ix_to_role'][index]=r
            V_dataset['role_to_ix'][r]=index
            index = index + 1

        V_dataset['ix_to_role'][index] = 'VERB'
        V_dataset['role_to_ix']['VERB'] = index

        for i in range(0,len(dataset['Q'])):
            for w in dataset['Q'][i].split():
                Vocabulary.add(w.lower())

        V_dataset['ix_to_word'] = {}
        V_dataset['word_to_ix'] = {}
        index = 0
        for v in Vocabulary:
            V_dataset['word_to_ix'][v] =  index
            V_dataset['ix_to_word'][index] = v
            index = index + 1

        V_dataset['word_to_ix']['unseen'] = index
        V_dataset['ix_to_word'][index] = 'unseen'

        counter = 0
        for index, w in sorted(V_dataset['ix_to_word'].items(), key=lambda p:p[1], reverse=True):
            counter = counter + 1
            # print( index, )

        V_dataset['ix_to_ans'] = {}
        V_dataset['ans_to_ix'] = {}

        index = 0
        for ans in top_1000_ans:
            # ans_index = V_dataset['word_to_ix'][ans]
            V_dataset['ix_to_ans'][index] = ans
            V_dataset['ans_to_ix'][ans] = index
            index = index + 1

        with open('{0}{1}'.format(C.data_dir, C.vocabulary_file), 'w') as fp:
            json.dump(V_dataset, fp)
    else:
        V_dataset = load_vocabulary()

    return V_dataset

def load_vocabulary():
    with open('{0}{1}'.format(C.data_dir, C.vocabulary_file), 'r') as fp:
        V_dataset = json.load(fp)
    return V_dataset

if __name__ == '__main__':
    with open('{0}{1}'.format(C.data_dir, C.vocabulary_file), 'r') as fp:
        V_dataset = json.load(fp)