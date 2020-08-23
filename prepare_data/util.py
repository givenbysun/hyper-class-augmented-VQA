
def chek_all_capital_letter(word):
    if len(word)==0:
        return False

    for c in word:
        if not c.isupper():
            return False
    return True

def extract_capital_words(words):
    result = []
    for word in words:
        if chek_all_capital_letter(word):
            result.append(word)
    return result

def filter_inside_paranthese(text):
    filtered_text = ''
    inside_paranthese = False
    for c in text:
        if c == '(':
            inside_paranthese = True
            continue
        if c == ')':
            inside_paranthese = False
            continue

        if not inside_paranthese:
            filtered_text = filtered_text + c

    return filtered_text

def filter_articles(words):
    filtedred_words = []
    for w in words:
        word = w.lower()
        if word =='a' or word =='an' or word =='the':
            continue
        filtedred_words.append(w)
    return filtedred_words


def recursive_question_construction(qbase, roles, role_prefix):
    n = len(roles)
    if n == 0:
        return [qbase]

    r = roles[0]
    roles.remove(r)
    q1 = recursive_question_construction(qbase, roles, role_prefix)
    q2 = recursive_question_construction(qbase + ' ' + role_prefix[r] + ' ' + r, roles, role_prefix)

    result = []
    for item in q1:
        result.append(item)
    for item in q2:
        result.append(item)

    return result
