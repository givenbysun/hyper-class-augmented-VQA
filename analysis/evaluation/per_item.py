

def read_file(file_name):
    f = open(file_name,'r')
    lines = f.readlines()
    items = []
    for l in lines:
        items.append(l.replace('\n',''))
    return items

base = 'baseline/wups/'
multi = 'multi/wups/'
answers = 'answers.txt'
predictions = 'predictions.txt'

base_answers = read_file(base+answers)
base_predictions = read_file(base+predictions)
multi_answers = read_file(multi+answers)
multi_predictions = read_file(multi+predictions)

val_questions = read_file('val_questions.txt')
val_roles = read_file('val_roles.txt')

def performance(answers, predictions, N):
    per_role_correct = {}
    per_qtype_correct = {}

    per_role_count = {}
    per_qtype_count = {}
    for i in range(N):
        w = val_questions[i].split()[0]
        r = val_roles[i]

        if w not in per_qtype_count:
            per_qtype_count[w] = 0

        if r not in per_role_count:
            per_role_count[r] = 0

        if w not in per_qtype_correct:
            per_qtype_correct[w] = 0

        if r not in per_role_correct:
            per_role_correct[r] = 0

        score = 0
        if answers[i]==predictions[i]:
            score = 1

        per_qtype_count[w] += 1
        per_role_count[r] += 1
        per_qtype_correct[w] += score
        per_role_correct[r] += score

    per_qtype_accuracy = {}
    per_role_accuracy = {}

    for w in per_qtype_count:
        per_qtype_accuracy[w] = per_qtype_correct[w]/float(per_qtype_count[w])

    for r in per_role_count:
        per_role_accuracy[r] = per_role_correct[r]/float(per_role_count[r])

    return per_qtype_accuracy, per_role_accuracy

N =len(val_questions)

base_per_qtype_accuracy, base_per_role_accuracy = performance(base_answers, base_predictions, N)
multi_per_qtype_accuracy, multi_per_role_accuracy = performance(multi_answers, multi_predictions, N)

for w in base_per_qtype_accuracy:
    print w, ' '*(20-len(w)), '\t' ,round(base_per_qtype_accuracy[w],2), '\t' ,round(multi_per_qtype_accuracy[w],2)

print

for r in base_per_role_accuracy:
    print r, ' '*(20-len(r)), '\t' ,round(base_per_role_accuracy[r],2), '\t' ,round(multi_per_role_accuracy[r],2)