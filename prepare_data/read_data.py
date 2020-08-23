import csv
import constants as C
import logging



def read_role_question_types():
    q_type = {}
    logging.info('@read_role_question_types')
    with open(C.data_dir + C.q_type_csv_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # print ' , '.join(row)
            q_type_key = row[0].strip()
            q_type[q_type_key]=[]
            q_type[q_type_key].append(row[1].strip())
            if len(row[2].strip())>0:
                if row[2].strip()=='+':
                    q_type[q_type_key].append(row[1].strip()+ ' ' + row[0].lower().strip())
                elif row[2].strip()=='*':
                    q_type[q_type_key].append(row[1].strip() + ' ' + row[0].lower().strip() + ' @use')
                else:
                    q_type[q_type_key].append(row[1].strip()+ ' ' + row[2].strip())

    return q_type


def read_general_semantic_class():
    role_class = {}
    logging.info('@read_general_semantic_class')
    with open(C.data_dir + C.r_class_csv_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # print ' , '.join(row)
            r_class_key = row[0].strip()

            classes = row[1].split('-')

            code = [0,0,0,0,0,0,0,0,0]
            for c in classes:
                c = c.strip()
                if c == 'AGENT':
                    code[0] = 1
                if c == 'COAGENT':
                    code[1] = 1
                if c == 'OBJECT':
                    code[2] = 1
                if c == 'ITEM':
                    code[3] = 1
                if c == 'PART':
                    code[4] = 1
                if c == 'TYPE':
                    code[5] = 1
                if c == 'TOOL':
                    code[6] = 1
                if c == 'LOCATION':
                    code[7] = 1
                if c == 'VERB':
                    code[8] = 1

            role_class[r_class_key] = code

    return role_class
