import utils
import torch
from torch.utils.data import DataLoader
from model import rc_cnn_dailmail
from torch.nn.parameter import Parameter
from train_eval import train
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='QA')
parser.add_argument('--input_size', default=100, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--train_num', default=100, type=int)
parser.add_argument('--train_batch_size', default=24, type=int)
parser.add_argument('--test_batch_size', default=12, type=int)
parser.add_argument('--train_date_path', default='./dateset/cnn/questions/training', type=str)
parser.add_argument('--test_date_path', default='./dateset/cnn/questions/test', type=str)
parser.add_argument('--glove_path', default='/nfs/users/guanxin/cache/.vector_cache', type=str)
config = parser.parse_args()

documents, questions, answers, doc_len, qus_len = utils.load_data(config.train_date_path, config.train_num, True)
test_documents, test_questions, test_answers, test_doc_len, test_qus_len = utils.load_data(config.test_date_path, 3000,
                                                                                           True)

# build word dict
word_dict = utils.build_dict(documents + questions)
embedding = Parameter(utils.embedding_word(word_dict, config.glove_path))

# build entity dict (numbers of categories)
entity_markers = list(set([w for w in word_dict.keys()
                           if w.startswith('@entity')] + answers))
entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}

doc_maxlen = max(map(len, (d.split(' ') for d in documents)))
query_maxlen = max(map(len, (q.split(' ') for q in questions)))

# data preprocessing, convert to one-hot
train_x1, train_x2, train_l, train_y = utils.vectorize(documents, questions, answers, word_dict, entity_dict,
                                                       doc_maxlen,
                                                       query_maxlen)

test_x1, test_x2, test_l, test_y = utils.vectorize(test_documents, test_questions, test_answers, word_dict, entity_dict,
                                                   doc_maxlen,
                                                   query_maxlen)

train_dataset = utils.NewsDataset(train_x1, train_x2, train_y, doc_len, qus_len)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)

test_dataset = utils.NewsDataset(test_x1, test_x2, test_y, doc_len, qus_len)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.test_batch_size, shuffle=True)

config.dict_num = max(word_dict.values()) + 1
config.eneity_num = len(entity_dict)
print('eneity class num: ', config.eneity_num)

model = rc_cnn_dailmail(config, embedding).cuda()
model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

train(config, train_loader, model, train_loader, optimizer)
