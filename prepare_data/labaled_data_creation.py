import json
import qa_temp_realization as qatr
import vocabulary_indexing as vi
import deepdish as dd
import constants as C
import logging
import os.path


def get_roles_and_answers(dataset):
    logging.info('@get_roles_and_answers')

    if os.path.isfile(C.data_dir+C.roles_and_top1000ans_file):
        with open('{0}{1}'.format(C.data_dir, C.roles_and_top1000ans_file), 'r') as fp:
            roles_and_answers = json.load(fp)
            return roles_and_answers['all_roles'], roles_and_answers['top_1000_ans']

    frequency_answers = {}
    frequency_roles = {}
    unique_answers = set([])
    set_of_all_roles = set([])
    for answer in dataset['A']:
        unique_answers.add(answer)
        if answer not in frequency_answers.keys():
            frequency_answers[answer] = 0
        frequency_answers[answer] = frequency_answers[answer] + 1
    for role in dataset['AR']:
        set_of_all_roles.add(role)
        if role not in frequency_roles.keys():
            frequency_roles[role] = 0
        frequency_roles[role] = frequency_roles[role] + 1

    counter = 0
    top_1000_ans = []
    for image_file_name, v in sorted(frequency_answers.items(), key=lambda p: p[1], reverse=True):
        counter = counter + 1
        # print(counter, (image_file_name, v))
        if counter <= 1000:
            top_1000_ans.append(image_file_name)

    with open('{0}{1}'.format(C.data_dir, C.roles_and_top1000ans_file), 'w') as fp:
        json.dump({'all_roles':list(set_of_all_roles), 'top_1000_ans':top_1000_ans}, fp)

    return set_of_all_roles, top_1000_ans


def label_data(role_class, phase='train'):
    logging.info('@label_data')
    qa_dataset = qatr.realize_imSitu_abstract_question_realization()[phase]
    set_of_all_roles, top_1000_ans = get_roles_and_answers(qa_dataset)
    V_dataset = vi.create_vocabulary(set_of_all_roles, top_1000_ans, phase)

    print V_dataset['role_to_ix'].keys()
    # print V_dataset['ix_to_role'][185]
    print len(V_dataset['role_to_ix'].keys())
    print len(V_dataset['ix_to_role'].keys())
    print len(V_dataset['ix_to_word'].keys())
    print len(V_dataset['word_to_ix'].keys())

    IQA_dataset = {}
    IQA_dataset['X'] = []
    X_image = []
    X_question = []
    X_role = []


    IQA_dataset['Y'] = []
    IQA_dataset['Yrole'] = []
    IQA_dataset['Ymulti'] = []
    IQA_dataset['I'] = []
    IQA_dataset['Q'] = []
    IQA_dataset['A'] = []
    IQA_dataset['QR'] = []
    IQA_dataset['AR'] = []

    counter = 0
    print 'ANSWER'

    role_answer_index = qatr.extract_role_asnwer_top1000_index(V_dataset, top_1000_ans)
    role_Y_top1000 = {}
    for role in set_of_all_roles:
        role = role.upper()
        role_Y_top1000[role] = []
        for j in range(0, 1000):
            role_Y_top1000[role].append(0)
        for answer_index in role_answer_index[role]:
            index = top_1000_ans.index(answer_index)
            role_Y_top1000[index] = 1

    Y_ans = {}
    for ans in V_dataset['ans_to_ix'].keys():
        Y_ans[ans] = []
        for j in range(0, 1000):
            Y_ans[ans].append(0)
        Y_ans['not_top_1000'] = Y_ans[ans]
        if ans in V_dataset['ans_to_ix'].keys():
            index = V_dataset['ans_to_ix'][ans]
            Y_ans[ans][index] = 1


    with open(C.data_dir + C.img_features_file, 'r') as json_file:
        image_data = json.load(json_file)
        for i in range(0, len(qa_dataset['Q'])):
            X = []
            R = []
            for j in range(0, 26):
                X.append(0)
                R.append(0)
            q_words = qa_dataset['Q'][i].split()
            q_roles = qa_dataset['QR'][i]

            for k in range(0, len(q_words)):
                if q_words[k].lower() not in V_dataset['word_to_ix'].keys():
                    X[26 - 1 - k] = V_dataset['word_to_ix']['unseen']
                    R[26 - 1 - k] = 0
                    continue
                X[26 - 1 - k] = V_dataset['word_to_ix'][q_words[k].lower()]
                # R[26-1-k] = V_dataset['role_to_ix'][q_roles[k]]+1
                if q_roles[k] in role_class.keys():
                    R[26 - 1 - k] = role_class[q_roles[k]]
                else:
                    R[26 - 1 - k] = [0, 0, 0, 0, 0, 0, 0, 0, 0]


            Y = []
            ans = qa_dataset['A'][i].lower()
            if ans in V_dataset['ans_to_ix'].keys():
                Y = Y_ans[ans]
            else:
                Y = Y_ans['not_top_1000']

            Y_role = []
            for j in range(0, len(V_dataset['role_to_ix'].keys())):
                Y_role.append(0)
            ans_role = qa_dataset['AR'][i].upper()
            # print '############### {0}'.format(len(set_of_all_roles))
            # print len(V_dataset['role_to_ix'].keys())
            # print len(V_dataset['ix_to_role'].keys())
            if ans_role in V_dataset['role_to_ix'].keys():
                index = V_dataset['role_to_ix'][ans_role]
                Y_role[index] = 1

            Y_multi = role_Y_top1000[ans_role]

            # X_image.append(get_image_representation(VGG_model, img_dir+dataset['I'][i]))
            X_image.append(image_data[phase][qa_dataset['I'][i]])
            X_question.append(X)
            X_role.append(R)
            IQA_dataset['Y'].append(Y)
            IQA_dataset['Yrole'].append(Y_role)
            IQA_dataset['Ymulti'].append(Y_multi)

            IQA_dataset['I'].append(qa_dataset['I'][i])
            IQA_dataset['Q'].append(qa_dataset['Q'][i])
            IQA_dataset['A'].append(qa_dataset['A'][i].lower())
            IQA_dataset['QR'].append(qa_dataset['QR'][i])
            IQA_dataset['AR'].append(qa_dataset['AR'][i].upper())

            counter = counter + 1

            if counter % 1000 == 0:
                print counter
                # break

    IQA_dataset['X'].append(X_image)
    IQA_dataset['X'].append(X_question)
    IQA_dataset['X'].append(X_role)

    dd.io.save('{0}vqa_{1}_info.h5'.format(C.data_dir, phase), IQA_dataset)

def perform():
    logging.basicConfig(level=logging.INFO)
    label_data(role_class={}, phase='train')
    # label_data(role_class={} ,phase='dev')
    label_data(role_class={}, phase='test')

if __name__ == '__main__':
    # perform()

    x = 0

