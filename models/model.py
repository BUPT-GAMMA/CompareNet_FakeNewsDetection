import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *
from torch.nn.parameter import Parameter
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce
import pickle as pkl

class HGAT(nn.Module):
    def __init__(self, params):
        super(HGAT, self).__init__()
        self.para_init()
        self.attention = True
        self.lower_attention = True

        # self.nonlinear = nn.LeakyReLU(0.2)
        self.nonlinear = F.relu_
        nfeat_list = [params.hidden_dim] * ({0: 1, 1: 2, 2: 2, 3: 3}[params.node_type])
        self.ntype = len(nfeat_list)
        nhid = params.node_emb_dim

        self.gc2: nn.ModuleList = nn.ModuleList()
        if not self.lower_attention:
            self.gc1: nn.ModuleList = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append( GraphConvolution(nfeat_list[t], nhid, bias=False) )
                self.bias1 = Parameter( torch.FloatTensor(nhid) )
                stdv = 1. / math.sqrt(nhid)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(nfeat_list, nhid, gamma=0.1)
        if self.attention:
            self.at1: nn.ModuleList = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append( SelfAttention(nhid, t, nhid // 2) )
        self.dropout = nn.Dropout(params.dropout)

    def para_init(self):
        self.attention = False
        self.lower_attention = False

    def forward(self, x_list, adj_list, adj_all = None):
        x0 = x_list

        if not self.lower_attention:
            x1 = [None for _ in range(self.ntype)]
            # 第一层gcn，与第一层后的dropout
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append( self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.dropout(self.nonlinear(x_t1))
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.dropout(self.nonlinear(x_t1))
                x1[t1] = x_t1

        return x1


    def inference(self, x_list, adj_list, adj_all = None):
        return self.forward(x_list, adj_list, adj_all)


class TextEncoder(Module):
    def __init__(self, params):
        super(TextEncoder, self).__init__()

        self.lstm = LstmEncoder(params.hidden_dim, params.emb_dim)

    def forward(self, embeds, seq_lens):
        return self.lstm(embeds, seq_lens)

class EntityEncoder(Module):
    def __init__(self, params):
        super(EntityEncoder, self).__init__()
        self.lstm = LstmEncoder(params.hidden_dim, params.emb_dim)
        self.gating = GatingMechanism(params)

    def forward(self, embeds, seq_lens, Y):
        X = self.lstm(embeds, seq_lens)
        return self.gating(X, Y)

class Pooling(nn.Module):
    def __init__(self, params):
        super(Pooling, self).__init__()
        self.mode = params.pooling
        self.params = params
        if self.mode == 'max':
            self.pooling = torch.max
        elif self.mode == 'sum':
            self.pooling = torch.sum
        elif self.mode == 'mean':
            self.pooling = torch.mean
        elif self.mode == 'att':
            self.pooling = AttentionPooling(self.params)
        else:
            raise Exception("Unknown pooling mode: {}. (Supported: max, sum, mean, att)".format(self.mode))

    def forward(self, X, sentPerDoc):
        '''
        :param X:           A tensor with shape:  (D1 + D2 + ... + Dn) * H
        :param sentPerDoc:  A tensor with values: [D1, D2, ..., Dn]
        :return:            A tensor with shape:  n * H
        '''
        # weight = [torch.ones((1, i.item()), device=sentPerDoc.device) for i in sentPerDoc]
        # weight = block_diag([m.to_sparse() for m in weight]).to_dense()
        sentPerDoc = sentPerDoc.cpu().numpy().tolist()
        sents = [X[sum(sentPerDoc[: i]): sum(sentPerDoc[: i+1])] for i in range(len(sentPerDoc))]
        output = []
        for s in sents:
            if s.shape[0] == 0:
                output.append(torch.zeros((1, s.shape[1]), device=s.device, dtype=X.dtype))
            else:
                cache = self.pooling(s, dim=0, keepdim=True)
                output.append(cache[0] if isinstance(cache, tuple) else cache)
        output = torch.cat(output, dim=0)
        return output

class ConcatTransform(nn.Module):
    # 分离试验用的
    def __init__(self, params):
        super(ConcatTransform, self).__init__()
        self.params = params
        self.preW = nn.Linear(self.params.hidden_dim, self.params.node_emb_dim)
        self.postW = nn.Linear(self.params.node_emb_dim * 2, self.params.node_emb_dim)
        self.dropout = nn.Dropout(self.params.dropout, )

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   shape: (N, node_emb_dim)
        :param Y:   shape: (N, hidden_dim)
        :return:    shape: (N, node_emb_dim)
        '''
        Y = self.preW(self.dropout(Y))                # (N, node_emb_dim)
        concatVector = torch.cat([X, Y], dim=1)            # (N, 2 * node_emb_dim)
        concatVector = self.postW(self.dropout(concatVector))
        return concatVector   # (N, node_emb_dim)

class MatchingTransform(nn.Module):
    def __init__(self, params):
        super(MatchingTransform, self).__init__()
        self.params = params
        self.SIMPLE = True
        self.preW = nn.Linear(self.params.hidden_dim, self.params.node_emb_dim)
        self.postW = nn.Linear(self.params.node_emb_dim * (2 if self.SIMPLE else 4), self.params.node_emb_dim)
        # self.nonlinear = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(self.params.dropout, )

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   shape: (N, node_emb_dim)
        :param Y:   shape: (N, hidden_dim)
        :return:    shape: (N, node_emb_dim)
        '''
        Y = self.preW(self.dropout(Y))                # (N, node_emb_dim)
        # if self.SIMPLE:     matchingVector = torch.cat([X - Y, X.mul(Y)], dim=1)            # (N, 2 * node_emb_dim)
        if self.SIMPLE:     matchingVector = torch.cat([X - Y, X.mul(Y)], dim=1)            # (N, 2 * node_emb_dim)
        else:               matchingVector = torch.cat([X, Y, X - Y, X.mul(Y)], dim=1)      # (N, 4 * node_emb_dim)
        matchingVector = self.postW(self.dropout(matchingVector))
        return matchingVector   # (N, node_emb_dim)

class GatingMechanism(nn.Module):
    def __init__(self, params):
        super(GatingMechanism, self).__init__()
        self.params = params
        with open(self.params.entity_tran, 'rb') as f:
            transE_embedding = pkl.load(f)
        self.enti_tran = nn.Embedding.from_pretrained(torch.from_numpy(transE_embedding).float())
        entity_num = transE_embedding.shape[0]
        # gating 的参数

        self.gate_theta = Parameter(torch.empty(entity_num, self.params.hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)

        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.LongTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta[Y])
        Y = self.enti_tran(Y)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output


if __name__ == '__main__':
    from main import parse_arguments
    GatingMechanism(parse_arguments())