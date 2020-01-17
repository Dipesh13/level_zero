# # 2 ipynb
# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": 1,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import pickle\n",
#     "import spacy\n",
#     "nlp = spacy.load('en_core_web_lg')"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 2,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "1667 2231\n"
#      ]
#     }
#    ],
#    "source": [
#     "train_set = None\n",
#     "with open(\"train.pkl\", \"rb\") as fi:\n",
#     "    train_set = pickle.load(fi)\n",
#     "\n",
#     "eval_set = None\n",
#     "with open(\"eval.pkl\", \"rb\") as fi:\n",
#     "    eval_set = pickle.load(fi)\n",
#     "\n",
#     "print (len(train_set), len(eval_set))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 3,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import numpy as np\n",
#     "import nltk\n",
#     "from nltk.corpus import stopwords\n",
#     "stops = stopwords.words(\"english\")"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 4,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def prepare(corpus):\n",
#     "    dataset = []\n",
#     "    for item, label in corpus:\n",
#     "        nouns = []\n",
#     "        verbs = []\n",
#     "        nouns_verbs = []\n",
#     "        words = [] \n",
#     "        tokens = nlp(item)\n",
#     "        for token in tokens:\n",
#     "            text = token.text.lower()\n",
#     "            lemma = token.lemma_\n",
#     "            if text in stops or lemma in stops:\n",
#     "                continue\n",
#     "            tag = token.tag_\n",
#     "            is_oov = token.is_oov\n",
#     "            has_vector = token.has_vector\n",
#     "            if not has_vector or is_oov:\n",
#     "                continue\n",
#     "            vector = token.vector\n",
#     "            if tag.startswith(\"NN\"):\n",
#     "                nouns.append(vector)\n",
#     "                nouns_verbs.append(vector)\n",
#     "            elif tag.startswith(\"VB\"):\n",
#     "                verbs.append(vector)\n",
#     "                nouns_verbs.append(vector)\n",
#     "            words.append(vector)        \n",
#     "        nouns = np.array(nouns)\n",
#     "        nouns = np.mean(nouns, axis=0, keepdims=True)        \n",
#     "        verbs = np.array(verbs)\n",
#     "        verbs = np.mean(verbs, axis=0, keepdims=True)\n",
#     "        nouns_verbs = np.array(nouns_verbs)\n",
#     "        nouns_verbs = np.mean(nouns_verbs, axis=0, keepdims=True)\n",
#     "        words = np.array(words)        \n",
#     "        words = np.mean(words, axis=0, keepdims=True)\n",
#     "        sample = {\n",
#     "            \"nouns\": nouns if len(nouns.shape) > 1 else None,\n",
#     "            \"verbs\": verbs if len(verbs.shape) > 1 else None,\n",
#     "            \"words\": words if len(words.shape) > 1 else None,\n",
#     "            \"nouns_verbs\": nouns_verbs if len(nouns_verbs.shape) > 1 else None,\n",
#     "            \"label\": label\n",
#     "        }\n",
#     "        dataset.append(sample)\n",
#     "    return dataset"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 5,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stderr",
#      "output_type": "stream",
#      "text": [
#       "/Users/mohitshah/.local/share/virtualenvs/in-out-rWorBeNE/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.\n",
#       "  out=out, **kwargs)\n",
#       "/Users/mohitshah/.local/share/virtualenvs/in-out-rWorBeNE/lib/python3.6/site-packages/numpy/core/_methods.py:73: RuntimeWarning: invalid value encountered in true_divide\n",
#       "  ret, rcount, out=ret, casting='unsafe', subok=False)\n"
#      ]
#     }
#    ],
#    "source": [
#     "train_data = prepare(train_set)\n",
#     "eval_data = prepare(eval_set)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 6,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def get_features(dataset, key=\"nouns\"):    \n",
#     "    items = [(x[key], x[\"label\"]) for x in dataset if x[key] is not None]\n",
#     "    x = [xx[0] for xx in items]\n",
#     "    y = [xx[1] for xx in items]\n",
#     "    x = np.array(x).squeeze()    \n",
#     "    return x, y"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 7,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "nouns_train_x, nouns_train_y = get_features(train_data, \"nouns\")\n",
#     "nouns_eval_x, nouns_eval_y = get_features(eval_data, \"nouns\")\n",
#     "verbs_train_x, verbs_train_y = get_features(train_data, \"verbs\")\n",
#     "verbs_eval_x, verbs_eval_y = get_features(eval_data, \"verbs\")\n",
#     "nv_train_x, nv_train_y = get_features(train_data, \"nouns_verbs\")\n",
#     "nv_eval_x, nv_eval_y = get_features(eval_data, \"nouns_verbs\")\n",
#     "words_train_x, words_train_y = get_features(train_data, \"words\")\n",
#     "words_eval_x, words_eval_y = get_features(eval_data, \"words\")"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 8,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "from sklearn.svm import OneClassSVM\n",
#     "def train_classifier(train_x):\n",
#     "    clf = OneClassSVM()\n",
#     "    clf.fit(train_x)\n",
#     "    return clf\n",
#     "\n",
#     "def get_predictions(clf, x):\n",
#     "    return clf.predict(x)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 9,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "n_clf = train_classifier(nouns_train_x)\n",
#     "v_clf = train_classifier(verbs_train_x)\n",
#     "nv_clf = train_classifier(nv_train_x)\n",
#     "w_clf = train_classifier(words_train_x)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 10,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "n_preds = get_predictions(n_clf, nouns_eval_x)\n",
#     "v_preds = get_predictions(v_clf, verbs_eval_x)\n",
#     "nv_preds = get_predictions(nv_clf, nv_eval_x)\n",
#     "w_preds = get_predictions(w_clf, words_eval_x)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 11,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "from sklearn.metrics import classification_report"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 12,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "         -1       0.70      0.28      0.40      1253\n",
#       "          1       0.41      0.81      0.54       765\n",
#       "\n",
#       "avg / total       0.59      0.48      0.45      2018\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "print (classification_report(n_preds, nouns_eval_y))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 13,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "         -1       0.50      0.26      0.34       572\n",
#       "          1       0.50      0.75      0.60       581\n",
#       "\n",
#       "avg / total       0.50      0.50      0.47      1153\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "print (classification_report(v_preds, verbs_eval_y))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 14,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "         -1       0.64      0.25      0.35      1398\n",
#       "          1       0.36      0.75      0.49       780\n",
#       "\n",
#       "avg / total       0.54      0.43      0.40      2178\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "print (classification_report(nv_preds, nv_eval_y))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 15,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "         -1       0.58      0.24      0.33      1373\n",
#       "          1       0.37      0.72      0.49       839\n",
#       "\n",
#       "avg / total       0.50      0.42      0.39      2212\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "print (classification_report(w_preds, words_eval_y))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 16,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "          0       0.39      0.13      0.19       498\n",
#       "          1       0.77      0.93      0.84      1520\n",
#       "\n",
#       "avg / total       0.67      0.73      0.68      2018\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "norm = np.linalg.norm(nouns_train_x, 2, axis=1).reshape(-1, 1)\n",
#     "n_t_x = nouns_train_x/norm\n",
#     "norm = np.linalg.norm(nouns_eval_x, 2, axis=1).reshape(-1, 1)\n",
#     "n_e_x = nouns_eval_x/norm\n",
#     "sims = np.dot(n_e_x, n_t_x.T)\n",
#     "max_sims = np.max(sims, axis=1)\n",
#     "preds = [int(x > 0.5) for x in max_sims]\n",
#     "truth = [int(x > 0) for x in nouns_eval_y]\n",
#     "print (classification_report(truth, preds))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 17,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "             precision    recall  f1-score   support\n",
#       "\n",
#       "          0       0.28      0.85      0.42       556\n",
#       "          1       0.83      0.26      0.39      1656\n",
#       "\n",
#       "avg / total       0.69      0.41      0.40      2212\n",
#       "\n"
#      ]
#     }
#    ],
#    "source": [
#     "norm = np.linalg.norm(words_train_x, 2, axis=1).reshape(-1, 1)\n",
#     "n_t_x = words_train_x/norm\n",
#     "norm = np.linalg.norm(words_eval_x, 2, axis=1).reshape(-1, 1)\n",
#     "n_e_x = words_eval_x/norm\n",
#     "sims = np.dot(n_e_x, n_t_x.T)\n",
#     "max_sims = np.max(sims, axis=1)\n",
#     "preds = [int(x > 0.85) for x in max_sims]\n",
#     "truth = [int(x > 0) for x in words_eval_y]\n",
#     "print (classification_report(truth, preds))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.6.5"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 2
# }