import torch
import torch.nn as nn
import torch.nn.functional as F


class rc_cnn_dailmail(nn.Module):
    def __init__(self, config, embedding):
        super(rc_cnn_dailmail, self).__init__()
        self.dict_embedding = nn.Embedding(num_embeddings=config.dict_num, embedding_dim=100, _weight=embedding)
        # self.dict_embedding.weight.requires_grad = False

        self.bilinear = nn.Bilinear(config.hidden_size * 2, config.hidden_size * 2, 1)
        self.lstm1 = nn.GRU(config.input_size, config.hidden_size, bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.lstm2 = nn.GRU(config.input_size, config.hidden_size, bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.linear = nn.Linear(config.hidden_size * 2, config.eneity_num)

    def forward(self, document, question, d_l, q_l):
        document = self.dict_embedding(document)
        question = self.dict_embedding(question)
        d_max_len = len(document[0])
        q_max_len = len(question[0])

        # reshape the input size (document) based on document sizes
        d_lengths, d_idx = d_l.sort(0, descending=True)
        _, d_un_idx = torch.sort(d_idx, dim=0)
        document = document[d_idx]
        d_packed_input = nn.utils.rnn.pack_padded_sequence(input=document, lengths=d_lengths.cpu(), batch_first=True)
        output1, _ = self.lstm1(d_packed_input)
        output1, _ = nn.utils.rnn.pad_packed_sequence(output1, batch_first=True, total_length=d_max_len)
        # change back to the original shape
        output1 = torch.index_select(output1, 0, d_un_idx)

        # ditto(question input)
        q_lengths, q_idx = q_l.sort(0, descending=True)
        _, q_un_idx = torch.sort(q_idx, dim=0)
        question = question[q_idx]
        q_packed_input = nn.utils.rnn.pack_padded_sequence(input=question, lengths=q_lengths.cpu(), batch_first=True)
        output2, last_hidden = self.lstm2(q_packed_input)
        output2, _ = nn.utils.rnn.pad_packed_sequence(output2, batch_first=True, total_length=q_max_len)
        output2 = torch.index_select(output2, 0, q_un_idx)

        # extract the last hidden of LSTM of question
        ques_hidden = [i - 1 for i in q_l]
        in_dim = [i for i in range(len(output2))]
        output2 = output2[in_dim, ques_hidden, :]
        output2 = output2.unsqueeze(dim=1)
        output2 = output2.repeat(1, len(document[1]), 1)

        # attention
        attn = self.bilinear(output1.contiguous(), output2)
        alpha = F.softmax(attn, dim=1)
        out = torch.sum(output1 * alpha, dim=1)

        out = self.linear(out)
        # out = F.relu(out)

        return out
