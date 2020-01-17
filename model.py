# from get_embedding import sent_embedding
# import en_core_web_sm
# nlp = en_core_web_sm.load()
import spacy
nlp = spacy.load('en')

print("\n ******** Summary for interest rate ********\n")

with open("/home/dipesh/auto_cca-master/test.txt") as f:
    sentences = f.readlines()

# SPACY
sentence1 = "interest rate"
token1 = nlp(sentence1.decode('utf8'))

X = []
for sentence in sentences:
    doc = nlp(sentence.decode('utf8'))
    for d in doc.sents:
        X.append((d.text.encode('utf8'),token1.similarity(d)))


def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup

X = Sort_Tuple(X)

for x in X[::-1]:
    if "$" in x[0]:
        print(x[0],x[1])


print("\n ******** Summary for foreign curreny exchange rate ********\n")

sentence2 = "operating income"
token2 = nlp(sentence2.decode('utf8'))

Y = []
for sentence in sentences:
    doc1 = nlp(sentence.decode('utf8'))
    for d1 in doc1.sents:
        Y.append((d1.text.encode('utf8'),token2.similarity(d1)))

Y = Sort_Tuple(Y)

for y in Y[::-1]:
    # if "operating" in y[0]:
    print(y[0],y[1])