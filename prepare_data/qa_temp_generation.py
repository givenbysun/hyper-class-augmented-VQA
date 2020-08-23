import json
import util
import read_data
import constants as C
import os.path
import logging



def create_imSitu_abstract_question_templates():
    logging.info('@create_imSitu_abstract_question_templates')

    if os.path.isfile('{0}{1}'.format(C.data_dir, C.qa_template_file)):
        return load_verb_answer_questions_template()

    verb_syntax = {}
    verb_role_prefix = {}
    verb_role_order = {}
    verb_answer_questions_template = {}
    set_of_all_roles = set([])


    data = json.load(open(C.data_dir + C.spec_data))
    with open(C.data_dir + C.abstracts_file, 'r') as f:
        abstracts = f.read().split('\n')
    q_type = read_data.read_role_question_types()

    abstract_index = 0
    for verb in data['verbs']:
        abstract = abstracts[abstract_index] # substituted ---> abstract = data['verbs'][verb]['abstract']
        abstract_index = abstract_index + 1

        if abstract.startswith('it is storming') \
                or abstract.startswith('it snow')\
                or abstract.startswith('it rains'):
            continue

        # abstract = abstract.replace(u"\u2019", " ")
        abstract = abstract.replace("'", " ")

        check_it = True

        words = util.filter_articles(util.filter_inside_paranthese(abstract.strip()).split())
        index = 0
        role_prefix = {}
        verb_role_prefix[verb]={}
        verb_answer_questions_template[verb]={}
        verb_role_order[verb] = []

        prefix = ''
        is_part_of_verb = []
        for word in words:
            is_part_of_verb.append(False)

        is_part_of_verb[1]= True

        if verb=='ailing': # exception
            is_part_of_verb[1] = False
            is_part_of_verb[2] = True

        if words[1] == 'is' or words[1] == 'to':
            is_part_of_verb[2] = True

        verb_syntax[verb] = ''

        for word in words:
            l = 3
            if verb[0:l] == word[0:l]:
                check_it = False
            if util.chek_all_capital_letter(word):
                role_prefix[word] = prefix
                verb_role_prefix[verb][word] = role_prefix[word]
                verb_role_order[verb].append(word)
                prefix = ''
            else:
                if not is_part_of_verb[index]:
                    prefix = prefix + ' ' + word

            index = index + 1

        verb_syntax[verb] = ''
        index = 0
        for word in words:
            if is_part_of_verb[index]:
                verb_syntax[verb] = verb_syntax[verb] + ' ' + word
            index = index + 1
        verb_syntax[verb] = verb_syntax[verb] + '#v'

        if verb=='ailing':
            x=0

        if check_it:
            print ' ERROR ' + verb
            print ' ABS ' + abstract


        AGENT = verb_role_order[verb][0]
        verb_answer_questions_template[verb]['VERB'] = []

        for role in verb_role_order[verb]:
            answer = str(role)
            set_of_all_roles.add(role.lower())
            verb_answer_questions_template[verb][answer] = []
            q_base = ''
            # simple_verb = '{0}'.format(verb[0:len(verb) - 3])

            simple_verb = (verb_syntax[verb].split())[-1]

            if answer.startswith(AGENT):
                q_base = 'who {0}'.format(verb_syntax[verb])
            elif answer == 'TOOL':
                q_base = 'what does {1} use to {0}'.format(simple_verb, AGENT)
            elif answer == 'PLACE':
                q_base = 'where does {1} {0}'.format(simple_verb, AGENT)
            else:
                if q_base.endswith('@use'):
                    q_base = '{0} does {2} use to {1}'.format(q_type[answer].replace('@use', ''), simple_verb, AGENT)
                else:
                    q_base = '{0} does {2} {1}'.format(q_type[answer][len(q_type[answer])-1], simple_verb, AGENT)

            q_roles = []
            for candidate_role in verb_role_order[verb]:
                if str(candidate_role) != AGENT:
                    if str(candidate_role) != answer:
                        q_roles.append(candidate_role)

            questions = util.recursive_question_construction(q_base, q_roles, role_prefix)



            # print abstract_2
            # print questions
            for q in questions:
                verb_answer_questions_template[verb][answer].append(q)

        q_base_action = 'what is {} doing'.format(AGENT)
        questions_action = []
        questions_action.append(q_base_action)
        if 'TOOL' in verb_role_order[verb]:
            questions_action.append('{0}{1}{2}'.format(q_base_action, role_prefix['TOOL'], ' TOOL'))
        if 'PLACE' in verb_role_order[verb]:
            questions_action.append('{0}{1}{2}'.format(q_base_action, role_prefix['PLACE'], ' PLACE'))
        if 'PLACE' in verb_role_order[verb] and 'TOOL' in verb_role_order[verb] :
            questions_action.append('{0}{1}{2}{3}{4}'.format(q_base_action, role_prefix['TOOL'], ' TOOL', role_prefix['PLACE'], ' PLACE'))
        for q2 in questions_action:
            verb_answer_questions_template[verb]['VERB'].append(q2)

        with open('{0}{1}'.format(C.data_dir, C.qa_template_file), 'w') as fp:
            json.dump(verb_answer_questions_template, fp)


    return verb_answer_questions_template


def load_verb_answer_questions_template():
    with open('{0}{1}'.format(C.data_dir, C.qa_template_file), 'r') as fp:
        verb_answer_questions_template = json.load(fp)
    return verb_answer_questions_template

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    verb_answer_questions_template = create_imSitu_abstract_question_templates()
    data = json.load(open(C.data_dir + C.spec_data))
    for verb in verb_answer_questions_template.keys():
        print("")
        print("")
        print(verb)
        print("")
        print(data['verbs'][verb]['abstract'])
        for answer in verb_answer_questions_template[verb]:
            print("")
            print(verb_answer_questions_template[verb][answer])
