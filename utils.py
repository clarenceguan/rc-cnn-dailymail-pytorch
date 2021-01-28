import os
import torchtext.vocab as vocab
from collections import Counter
import torch
import numpy as np
from torch.utils.data import Dataset


def load_data(files, max_example=None, relabeling=True):
    documents = []
    questions = []
    answers = []
    num_examples = 0
    doc_len = []
    qus_len = []
    train_files = os.listdir(files)
    for i, file in enumerate(train_files):
        if i < max_example:
            path = os.path.join(files, file)
            f = open(path, 'r')
            lines = f.readlines()

            document = lines[2].strip().lower()
            question = lines[4].strip().lower()
            answer = lines[6].strip()
            q_l = question.split(' ').__len__()
            d_l = document.split(' ').__len__()

            if relabeling:
                q_words = question.split(' ')
                d_words = document.split(' ')
                assert answer in d_words

                entity_id = 0
                entity_dict = {}

                for word in d_words + q_words:
                    if (word.startswith('@entity')) and (word not in entity_dict):
                        entity_dict[word] = '@entity' + str(entity_id)
                        entity_id += 1

                q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
                d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
                answer = entity_dict[answer]

                question = ' '.join(q_words)
                document = ' '.join(d_words)

            doc_len.append(d_l)
            qus_len.append(q_l)
            questions.append(question)
            answers.append(answer)
            documents.append(document)
            num_examples += 1
            f.close()
        else:
            break
    return documents, questions, answers, doc_len, qus_len


def build_dict(sentences, max_words=50000):
    # build_dict of this dateset
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def embedding_word(word_dict, embedding_path):
    glove = vocab.GloVe(name='6B', dim=100, cache='/nfs/users/guanxin/cache/.vector_cache')

    num_words = max(word_dict.values()) + 1
    embeddings = torch.zeros(num_words, 100)

    for word, i in word_dict.items():
        if word in glove.stoi:
            embeddings[i] = glove.vectors[glove.stoi[word]]

    return embeddings


def vectorize(documents, questions, answers, word_dict, entity_dict, doc_maxlen, q_maxlen):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(documents), len(entity_dict))).astype(float)
    in_y = []
    for idx, (d, q, a) in enumerate(zip(documents, questions, answers)):
        d_words = d.split(' ')
        q_words = q.split(' ')

        seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
        seq1 = seq1[:doc_maxlen]
        pad_1 = max(0, doc_maxlen - len(seq1))
        seq1 += [0] * pad_1
        seq1 = np.array(seq1)

        seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
        seq2 = seq2[:q_maxlen]
        pad_2 = max(0, q_maxlen - len(seq2))
        seq2 += [0] * pad_2
        seq2 = np.array(seq2)

        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0

            in_y.append(entity_dict[a] if a in entity_dict else 0)

    return in_x1, in_x2, in_l, in_y


class NewsDataset(Dataset):
    def __init__(self, document, question, answer, doc_len, qus_len):
        self.docoument = document
        self.question = question
        self.answer = answer
        self.doc_len = doc_len
        self.qus_len = qus_len

    def __len__(self):
        return len(self.docoument)

    def __getitem__(self, index):
        return self.docoument[index], self.question[index], self.answer[index], self.doc_len[index], self.qus_len[index]


def collate_data(batch):
    d = torch.tensor([item[0] for item in batch])
    q = torch.tensor([item[1] for item in batch])
    a = torch.tensor([item[2] for item in batch])

    d_l = torch.tensor([item[3] for item in batch])
    q_l = torch.tensor([item[4] for item in batch])

    return d, q, a, d_l, q_l
