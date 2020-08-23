import re
import json
import util
import constants as C
import logging
import deepdish as dd
import os.path
import qa_temp_generation as qtg

def realize_imSitu_abstract_question_realization():
    logging.info('@realize_imSitu_abstract_question_realization')

    if os.path.isfile('{0}{1}'.format(C.data_dir, C.qa_dataset_file)):
        return load_qa_dataset()

    data = json.load(open(C.data_dir + C.spec_data))
    qa_dataset = {}

    for phase in ['train','dev','test']:
        phase_data = json.load(open('{}/{}.json'.format(C.data_dir, phase)))
        verb_answer_questions_template = qtg.create_imSitu_abstract_question_templates()

        dataset = {}
        dataset['Q'] = []
        dataset['A'] = []
        dataset['R'] = []
        dataset['I'] = []
        dataset['V'] = []
        dataset['QR'] = []
        dataset['AR'] = []

        role_answers = {}

        for image_file_name in phase_data.keys():

            # print k
            # print train[k]
            searchObj = re.search(r'([a-zA-Z]+)_', image_file_name)
            V = searchObj.group(1).strip()
            if V == 'raining' or V == 'snowing' or V == 'storming':
                continue

            for A in verb_answer_questions_template[V].keys():
                for Q in verb_answer_questions_template[V][A]:
                    # print '{0} ---> {1}'.format(Q,A)

                    for f in phase_data[image_file_name]['frames']:
                        question = Q.strip()
                        answer = A
                        answer_role = A
                        question_words = question.split()
                        qa_roles = util.extract_capital_words(question_words)
                        qa_roles.append(answer)
                        # dataset['R'].append(str(answer))
                        frame_satisfies_q_roles = True
                        filled_roles = []

                        roles = f.keys()
                        roles = sorted(roles, key=len, reverse=True)

                        for role in roles:
                            if role != u'':
                                name = f[role]
                                if name != u'':
                                    filled_roles.append(role.upper())
                                    # print str(role).upper() , question_roles , str(role).upper() not in question_roles
                                    if str(role).upper() not in qa_roles:
                                        frame_satisfies_q_roles = False

                        for qa_role in qa_roles:
                            if qa_role not in filled_roles:
                                frame_satisfies_q_roles = False

                        if not frame_satisfies_q_roles:
                            continue

                        # print question_words

                        capacity = 1
                        question_template = question
                        q_role_value_dic = {}
                        for role in roles:
                            if role != u'':
                                name = f[role]
                                if name != u'':
                                    # unique_answers.add(str(name))
                                    role = str(role)
                                    gloss = data['nouns'][name]
                                    name = str(gloss[u'gloss'][0])
                                    capacity = capacity * len(gloss[u'gloss'])
                                    q_role_value_dic[role.upper()] = name
                                    new_question = ''
                                    for word in question.split():
                                        if role.upper() == word:
                                            new_question = new_question + name
                                        else:
                                            new_question = new_question + word
                                        new_question = new_question + ' '
                                    question = new_question.strip()
                                    # question = question.replace(role.upper(), name)

                                    answer = answer.replace('VERB', V)
                                    answer = answer.replace(role.upper(), name)
                                    if role not in role_answers.keys():
                                        role_answers[role] = {}
                                    if answer not in role_answers[role].keys():
                                        role_answers[role][answer] = 0
                                    role_answers[role][answer] = role_answers[role][answer] + 1
                        # summer = summer + capacity
                        # print question,' : ', answer

                        q_text_roles = []
                        for word in question_template.split():
                            if util.chek_all_capital_letter(word):
                                freq = len(q_role_value_dic[word].split())
                                for i in range(freq):
                                    q_text_roles.append(word)
                            elif word.endswith('#v'):
                                q_text_roles.append('VERB')
                            else:
                                q_text_roles.append('')
                        question = question.replace('#v', '')
                        if len(q_text_roles) != len(question.split()):
                            print question, len(question.split())
                            print q_text_roles, len(q_text_roles)
                            print Q
                            # raise Exception('ERROR! len(q_text_roles) != len(question.split())')

                        if len(answer)==0 or util.chek_all_capital_letter(answer):
                            print 'INVALID => Q:{0} A:{1}'.format(question, answer)
                            continue

                        dataset['QR'].append(q_text_roles)
                        dataset['Q'].append(question)
                        dataset['AR'].append(answer_role)
                        dataset['A'].append(answer)
                        dataset['V'].append(V)
                        dataset['I'].append(image_file_name)

        qa_dataset[phase] = dataset

        if phase=='train':
            role_asnwer_frequency = extract_role_asnwer_frequency(dataset)
            with open('{0}{1}'.format(C.data_dir, C.role_asnwer_frequency_file), 'w') as fp:
                json.dump(role_asnwer_frequency, fp)

    dd.io.save('{0}{1}'.format(C.data_dir, C.qa_dataset_file), qa_dataset)

    return qa_dataset

def extract_role_asnwer_top1000_index(V_dataset, top_1000_ans):
    role_answer_index = {}
    with open('{0}{1}'.format(C.data_dir, C.role_asnwer_frequency_file)) as fp:
        role_asnwer_frequency = json.load(fp)
        for role in role_asnwer_frequency.keys():
            role_answer_index[role.upper()] = []
            for answer in role_asnwer_frequency[role].keys():
                if answer not in V_dataset['ans_to_ix'].keys():
                    continue
                answer_index = V_dataset['ans_to_ix'][answer]
                if answer_index in top_1000_ans:
                    role_answer_index[role.upper()].append(answer_index)

    return role_answer_index

def extract_role_asnwer_frequency(dataset):
    number_of_samples = len(dataset['Q'])
    role_asnwer_frequency = {}
    for i in range(number_of_samples):
        role = dataset['AR'][i].lower()
        answer = dataset['A'][i].lower()
        if role not in role_asnwer_frequency.keys():
            role_asnwer_frequency[role] = {}
        if answer not in role_asnwer_frequency[role].keys():
            role_asnwer_frequency[role][answer] = 0
        role_asnwer_frequency[role][answer] = role_asnwer_frequency[role][answer] + 1

    return role_asnwer_frequency

def load_qa_dataset():
    file_name = '{0}{1}'.format(C.data_dir, C.qa_dataset_file)
    qa_dataset = dd.io.load(file_name)
    return qa_dataset

def realized_qa_analysis():
    dataset = realize_imSitu_abstract_question_realization()
    # dataset, role_answers = realize_imSitu_abstract_question_realization()
    # for r in role_answers:
    #     print
    #     print 'ROLE: ' + r
    #     sorted_r_a = []
    #     for A, a in sorted(role_answers[r].items(), key=lambda p: p[1], reverse=True):
    #         sorted_r_a.append('{0}:{1} '.format(A, a))
    #     print sorted_r_a

    print(dataset['train']['Q'][0])
    print(dataset['train']['A'][0])
    print(len(dataset['train']['Q']))
    print(len(dataset['dev']['Q']))
    print(len(dataset['test']['Q']))
    unique_answers = set([])
    frequency_answers = {}
    frequency_roles = {}
    frequency_answer_roles = {}
    for answer in dataset['train']['A']:
        unique_answers.add(answer)
        if answer not in frequency_answers.keys():
            frequency_answers[answer] = 0

        frequency_answers[answer] = frequency_answers[answer] + 1
    for role in dataset['train']['R']:
        if role not in frequency_roles.keys():
            frequency_roles[role] = 0
            frequency_roles[role] = frequency_roles[role] + 1

    for ra in dataset['train']['AR']:
        if ra not in frequency_answer_roles.keys():
            frequency_answer_roles[ra] = 0
        frequency_answer_roles[ra] = frequency_answer_roles[ra] + 1

    for r, v in sorted(frequency_answer_roles.items(), key=lambda p: p[1], reverse=True):
        print '{0} : {1}'.format(r,v)

    counter = 0
    top_1000_ans = []
    for image_file_name, v in sorted(frequency_answers.items(), key=lambda p: p[1], reverse=True):
        counter = counter + 1
        # print(counter, (image_file_name, v))
        if counter <= 10:
            top_1000_ans.append(image_file_name)
            print image_file_name, v

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = realize_imSitu_abstract_question_realization()
    # realized_qa_analysis()
    print len(dataset['train']['I'])
    k =0
    distinct = set([])
    phase='dev'
    for phase in ['train','test','dev']:
        for i in range(len(dataset[phase]['Q'])):
            if dataset[phase]['I'][i].startswith('catching') or True:
                # print k+1
                k+=1
                distinct.add(dataset[phase]['I'][i])
                # print dataset[phase]['I'][i]
            if dataset[phase]['I'][i]=='catching_24.jpg':
                print(dataset[phase]['Q'][i])
                print(dataset[phase]['A'][i])
                print(dataset[phase]['AR'][i])

    print distinct
    print len(distinct)

    data = json.load(open(C.data_dir + C.spec_data))
    qa_dataset = {}
    distinct = set([])
    for phase in ['train', 'dev', 'test']:
        phase_data = json.load(open('{}/{}.json'.format(C.data_dir, phase)))
        verb_answer_questions_template = qtg.create_imSitu_abstract_question_templates()

        role_answers = {}

        for image_file_name in phase_data.keys():
            # print k
            # print train[k]
            searchObj = re.search(r'([a-zA-Z]+)_', image_file_name)
            V = searchObj.group(1).strip()

            if V == 'raining' or V == 'snowing' or V == 'storming':
                continue

            for A in verb_answer_questions_template[V].keys():
                for Q in verb_answer_questions_template[V][A]:
                    # print '{0} ---> {1}'.format(Q,A)

                    for f in phase_data[image_file_name]['frames']:
                        question = Q.strip()
                        answer = A
                        distinct.add(image_file_name)
                        if image_file_name=='catching_24.jpg':
                            # if image_file_name == 'opening_251.jpg':
                            print image_file_name, question, answer
                            print image_file_name, question, answer
                            roles = f.keys()
                            roles = sorted(roles, key=len, reverse=True)

                            for role in roles:
                                name = f[role]
                                print role , name


    # print distinct
    print len(distinct)