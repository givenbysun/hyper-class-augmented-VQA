import re
import json
from prepare_data import util
import constants as C
import logging
import deepdish as dd
import os.path
from prepare_data import qa_temp_generation as qtg
from prepare_data import qa_temp_realization as qtr
import word_cloud_generation as wcg

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def generate_imSituVQA_word_cloud():
    qa_dataset = qtr.load_qa_dataset()
    # <type 'list'>: ['A', 'QR', 'I', 'Q', 'R', 'V', 'AR']
    f = open('tmp.txt', 'w')
    f.writelines(' '.join(qa_dataset['train']['Q']))
    # f.writelines(' '.join([' '.join(x) for x in qa_dataset['train']['QR']]))
    f.close()
    f = open('tmp.txt', 'r')
    wcg.generate_word_cloud(f.read(), 'cloud')
    return

def check_imSitu_qa_dataset_stat():
    qa_dataset = qtr.load_qa_dataset()

    length_counter = {}
    length_counter = {}
    total_words = 0
    length_counter_percent = {}

    first_q_word_counter = {}
    first_q_word_counter_all = 0
    second_q_word_counter = {}
    second_q_word_counter_all = 0

    templates_per_verb_counter = {}
    total_templates = 0



    print 'total templates are ', total_templates
    sorted_tpv = sorted(templates_per_verb_counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print sorted_tpv

    for phase in ['train', 'dev', 'test']:
        phase_data = qa_dataset[phase]

        role_answers = {}

        for Q, A in zip(phase_data['Q'], phase_data['A']):

                first_q_word = Q.split()[0]
                if first_q_word not in first_q_word_counter:
                    first_q_word_counter[first_q_word] = 0
                first_q_word_counter[first_q_word] += 1
                first_q_word_counter_all += 1

                second_q_word = Q.split()[1]
                if first_q_word == 'what':
                    if second_q_word not in second_q_word_counter:
                        second_q_word_counter[second_q_word] = 0
                    second_q_word_counter[second_q_word] += 1

                q = Q.split()
                ql = len(q)
                if ql not in length_counter:
                    length_counter[ql] = 0
                length_counter[ql] += 1
                total_words += 1

    print length_counter

    for l in length_counter:
        rf = length_counter[l] / float(total_words)
        length_counter_percent[l] = int(rf * 10000) / float(100)
    print 'length_counter_percent'
    for lcp in length_counter_percent:
        print '({0},{1})'.format(lcp, length_counter_percent[lcp])

    print sorted(first_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)
    print [c * 100 / float(first_q_word_counter_all) for w, c in
           sorted(first_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)]

    print sorted(second_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)
    print [c * 100 / float(first_q_word_counter['what']) for w, c in
           sorted(second_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)]


def check_imSitu_abstract_qa_templates_stat():
    # if os.path.isfile('{0}{1}'.format(C.data_dir, C.qa_dataset_file)):
    #     return load_qa_dataset()

    length_counter = {}
    length_counter = {}
    total_words = 0
    length_counter_percent = {}

    first_q_word_counter = {}
    first_q_word_counter_all = 0
    second_q_word_counter = {}
    second_q_word_counter_all = 0

    templates_per_verb_counter = {}
    total_templates = 0
    verb_answer_questions_template = qtg.create_imSitu_abstract_question_templates()
    for v in verb_answer_questions_template:
        templates_per_verb_counter[v] = 0
        for ar in verb_answer_questions_template[v]:
            # print verb_answer_questions_template[v][ar]
            number = len(verb_answer_questions_template[v][ar])
            total_templates += number
            templates_per_verb_counter[v] += number


    print 'total templates are ', total_templates
    sorted_tpv =  sorted(templates_per_verb_counter.items(),key=lambda kv:(kv[1], kv[0]) ,reverse=True)
    print sorted_tpv

    for phase in ['train','dev','test']:
        phase_data = json.load(open('{}/{}.json'.format(C.data_dir, phase)))

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

                    first_q_word = Q.split()[0]
                    if first_q_word not in first_q_word_counter:
                        first_q_word_counter[first_q_word] = 0
                    first_q_word_counter[first_q_word] +=1
                    first_q_word_counter_all += 1

                    second_q_word = Q.split()[1]
                    if first_q_word =='what':
                        if second_q_word not in second_q_word_counter:
                            second_q_word_counter[second_q_word] = 0
                        second_q_word_counter[second_q_word] +=1


                    q = Q.split()
                    ql = len(q)
                    if ql not in length_counter:
                        length_counter[ql] = 0
                    length_counter[ql] += 1
                    total_words +=1

    print length_counter

    for l in length_counter:
        rf = length_counter[l]/float(total_words)
        length_counter_percent[l] = int(rf*10000)/float(100)

    print 'length_counter_percent'
    for lcp in length_counter_percent:
        print '({0},{1})'.format(lcp, length_counter_percent[lcp])

    print sorted(first_q_word_counter.items(), key= lambda (x,y):(y,x) ,reverse=True)
    print [c*100 / float(first_q_word_counter_all) for w, c in
           sorted(first_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)]

    print sorted(second_q_word_counter.items(), key= lambda (x,y):(y,x) ,reverse=True)
    print [c*100 / float(first_q_word_counter['what']) for w, c in
           sorted(second_q_word_counter.items(), key=lambda (x, y): (y, x), reverse=True)]


if __name__ == '__main__':
    check_imSitu_abstract_qa_templates_stat()
    check_imSitu_qa_dataset_stat()
    generate_imSituVQA_word_cloud()