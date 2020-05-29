#!/user/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce
from models.model import HGAT, TextEncoder, EntityEncoder, Pooling, MatchingTransform, GatingMechanism
import pickle as pkl

class Classifier(nn.Module):
    def __init__(self, params, vocab_size, pte=None):
        super(Classifier, self).__init__()
        self.params = params
        self.vocab_size = vocab_size
        self.pte = False if pte is None else True

        self.text_encoder = TextEncoder(params)
        self.enti_encoder = EntityEncoder(params)
        # numOfEntity = 100000
        # self.enti_encoder = nn.Embedding(numOfEntity, params.hidden_dim)
        # nn.init.xavier_uniform_(self.enti_encoder.weight)
        self.topi_encoder = nn.Embedding(100, 100)
        self.topi_encoder.from_pretrained(torch.eye(100))
        self.match_encoder = MatchingTransform(params)
        # self.match_encoder = ConcatTransform(params)   # 参数试验用的
        self.word_embeddings = nn.Embedding(vocab_size, params.emb_dim)
        if pte is None:
            nn.init.xavier_uniform_(self.word_embeddings.weight)
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pte))
        # KB Field

        # with open(self.params.entity_tran, 'rb') as f:
        #     transE_embedding = pkl.load(f)
        # self.enti_tran = nn.Embedding.from_pretrained(torch.from_numpy(transE_embedding))

        self.model = HGAT(params)
        self.pooling = Pooling(params)
        self.classifier_sen = nn.Linear(params.node_emb_dim, params.ntags)
        self.classifier_ent = nn.Linear(params.node_emb_dim, params.ntags)

        self.dropout = nn.Dropout(params.dropout, )

        # entity_num = transE_embedding.shape[0]
        # self.gating = GatingMechanism(params) # 这个要放在最后面，尽量少影响随机初始化

    # def forward(self, x_list, adj_list, sentPerDoc, entPerDoc=None):
    def forward(self, documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc=None):
        x_list = []
        embeds_docu = self.word_embeddings(documents)   # sents * max_seq_len * emb
        d = self.text_encoder(embeds_docu, doc_lens)    # sents * max_seq_len * hidden
        d = self.dropout(F.relu_(d))                     # Relu activation and dropout
        x_list.append(d)
        if self.params.node_type == 3 or self.params.node_type == 2:
            embeds_enti = self.word_embeddings(ent_desc)    # sents * max_seq_len * emb
            e = self.enti_encoder(embeds_enti, ent_lens, feature_lists[1])    # sents * max_seq_len * hidden
            e = self.dropout(F.relu_(e))                     # Relu activation and dropout
            x_list.append(e)
        if self.params.node_type == 3 or self.params.node_type == 1:
            t = self.topi_encoder(feature_lists[-1])         # tops * hidden
            x_list.append(t)

        X = self.model(x_list, adj_lists)

        X_s = self.pooling(X[0], sentPerDoc)   # 选择句子的部分
        output = self.classifier_sen(X_s)

        if entiPerDoc is not None:
            # E_trans = self.enti_tran(feature_lists[1])
            E_GCN = X[1]
            # E_KB = self.gating(x_list[1], feature_lists[1])
            E_KB = x_list[1]
            X_e = self.match_encoder(E_GCN, E_KB)  # 选择实体的部分
            X_e = self.pooling(X_e, entiPerDoc)
            X_e = self.classifier_ent(X_e)
            output += X_e
        output = F.softmax(output, dim=1)       # 单分类
        # output = torch.sigmoid(output)        # 多分类
        return output


if __name__ == '__main__':
    pass