import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize, word_tokenize
from multiprocessing import Pool as ProcessPool

ASYMMETRIC = True
DEBUG_NUM = 400
W2I = None


def sentence_tokenize(doc):
    # return doc.split('.')
    return sent_tokenize(doc)

def read_and_unpkl(file):
    with open(file, 'rb') as f:
        return pkl.load(f)

def parseLine(args):
    idx, tag, doc = args
    global W2I
    # sentences = doc.split('.')
    sentences = sentence_tokenize(doc)
    sentences_idx = []
    for sentence in sentences:
        sentence = sentence.lower().strip().split(" ")
        curr_sentence_idx = [W2I[x] for x in sentence]
        sentences_idx.append(curr_sentence_idx if len(curr_sentence_idx) > 0 else [W2I['<unk>']])
    return int(tag), sentences_idx

class DataLoader:
    def __init__(self, params):
        self.params = params
        self.ntags = params.ntags

        train_pkl_path = '{}/train/'.format(params.adjs)
        test_pkl_path = '{}/test/'.format(params.adjs)
        dev_pkl_path = '{}/dev/'.format(params.adjs)
        print('Loading adj: ', train_pkl_path[: -6])

        w2i_pkl_path = params.root + 'w2i.pkl'
        if params.mode == 0:
            w2i = freezable_defaultdict(lambda: len(w2i))
            UNK = w2i["<unk>"]

            self.train, self.adj_train, self.fea_train = self.read_dataset(params.train, w2i, train_pkl_path)
            print("Average train document length: {}".format(np.mean([len(x[0]) for x in self.train])))
            print("Maximum train document length: {}".format(max([len(x[0]) for x in self.train])))

            self.train, self.dev, self.adj_train, self.adj_dev, self.fea_train, self.fea_dev = \
                train_test_split(self.train, self.adj_train, self.fea_train, test_size=0.2, random_state=42)
        else:
            with open(w2i_pkl_path, 'rb') as f:
                w2i = pkl.load(f)
                UNK = w2i["<unk>"]

        w2i = freezable_defaultdict(lambda: UNK, w2i)
        w2i.freeze()
        self.w2i = w2i
        self.i2w = dict(map(reversed, self.w2i.items()))
        self.nwords = len(w2i)


        with open(params.entity_desc, 'rb') as f:
            corpus = pkl.load(f)
        self.entity_description = []
        for row in corpus:
            self.entity_description.append([w2i[x] for x in row.lower().split(" ")])

        if params.mode == 0:
            dataset_train = DataSet(self.train, self.adj_train, self.fea_train, self.params, self.entity_description)
            self.train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                    batch_size=params.batch_size, collate_fn=dataset_train.collate, shuffle=True)
            dataset_dev = DataSet(self.dev, self.adj_dev, self.fea_dev, self.params, self.entity_description)
            self.dev_data_loader = torch.utils.data.DataLoader(dataset_dev,
                                    batch_size=params.batch_size, collate_fn=dataset_dev.collate,   shuffle=False)


        self.test, self.adj_test, self.fea_test = self.read_dataset(params.test, w2i, test_pkl_path)
        self.test_2, self.adj_test_2, self.fea_test_2 = self.read_dataset(params.dev, w2i, dev_pkl_path)

        dataset_test = DataSet(self.test, self.adj_test, self.fea_test, self.params, self.entity_description)
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                batch_size=params.batch_size, collate_fn=dataset_test.collate,  shuffle=False)
        dataset_test_2 = DataSet(self.test_2, self.adj_test_2, self.fea_test_2, self.params, self.entity_description)
        self.test_data_loader_2 = torch.utils.data.DataLoader(dataset_test_2,
                                batch_size=params.batch_size, collate_fn=dataset_test_2.collate,shuffle=False)


    def load_adj_and_other(self, path):
        print("Loading {}".format(path))
        if path[-1] == '/':
            files = sorted([path + f for f in os.listdir(path) if judge_data(f)],
                                key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
            files = files[: DEBUG_NUM] if self.params.DEBUG else files
            data = [read_and_unpkl(file) for file in tqdm(files)]
        else:
            with open(path, 'rb') as f:
                data = pkl.load(f)
        print("Preprocessing {}".format(path))
        res, device = [], 'cuda' if self.params.cuda else 'cpu'
        for piece in tqdm(data):
            d_idx = piece['idx']
            adj_list = [build_spr_coo(a) for a in piece['adj_list']]
            feature_list = [piece['s2i'], piece['e2i'], piece['t2i']]
            res.append([adj_list, feature_list])
        return res

    def read_dataset(self, filename, w2i, adj_file):
        adj = self.load_adj_and_other(adj_file)
        if 'csv' in filename:
            return self.read_dataset_sentence_wise(filename, w2i, adj)
        if 'xlsx' in filename:
            return self.read_testset_sentence_wise(filename, w2i, adj)

    def read_dataset_sentence_wise(self, filename, w2i, adj):
        data, new_adj, new_fea, removed_idx = [], [], [], []
        global W2I
        W2I = w2i
        # count = 0
        adj, fea = zip(*adj)
        with open(filename, "r") as f:
            readCSV = csv.reader(f, delimiter=',')
            csv.field_size_limit(100000000)
            sents = []
            for idx, (tag, doc) in tqdm(enumerate(readCSV)):
                if self.params.DEBUG and idx >= DEBUG_NUM:
                    break
                sents.append([idx, tag, doc])

            sentences_idx_list = []
            p = ProcessPool(10)
            with tqdm(total=len(sents)) as pbar:
                for out in p.imap(parseLine, sents):
                    sentences_idx_list.append(out)
                    pbar.update(1)
            p.close()
            p.join()

            print(len(sentences_idx_list))
            allowed_tags = [1, 4] if self.ntags == 2 else [1, 2, 3, 4]
            for idx, (tag, sentences_idx) in enumerate(sentences_idx_list):
                if tag in allowed_tags:
                    if self.ntags == 2:
                        tag = tag - 1 if tag == 1 else tag - 3   # Adjust the tag to {0: Satire, 1: Trusted}
                    else:
                        tag -= 1                                 # {0: Satire, 1: Hoax, 2: Propaganda, 3: Trusted}
                    if len(sentences_idx) > 1:
                        data.append((sentences_idx[:self.params.max_sents_in_a_doc], tag))
                        new_adj.append(adj[idx])
                        new_fea.append(fea[idx])
                    else:
                        removed_idx.append(idx)
        print('removed_idx of {}: {}'.format(filename, len(removed_idx)))
        print(len(data), len(new_adj))
        return data, new_adj, new_fea

    def read_dataset_sentence_wise(self, filename, w2i, adj):
        data, new_adj, new_fea = [], [], []
        # count = 0
        adj, fea = zip(*adj)
        with open(filename, "r") as f:
            readCSV = csv.reader(f, delimiter=',')
            csv.field_size_limit(100000000)
            removed_idx = []
            for idx, (tag, doc) in tqdm(enumerate(readCSV)):
                if self.params.DEBUG and idx >= DEBUG_NUM:
                    break
                # sentences = doc.split('.')
                sentences = sentence_tokenize(doc)
                tag = int(tag)
                allowed_tags = [1, 4] if self.ntags == 2 else [1, 2, 3, 4]
                if tag in allowed_tags:
                    if self.ntags == 2:
                        tag = tag - 1 if tag == 1 else tag - 3   # Adjust the tag to {0: Satire, 1: Trusted}
                    else:
                        tag -= 1                                 # {0: Satire, 1: Hoax, 2: Propaganda, 3: Trusted}
                    sentences_idx = []
                    for sentence in sentences:
                        sentence = sentence.lower().strip().split(" ")
                        curr_sentence_idx = [w2i[x] for x in sentence]
                        sentences_idx.append(curr_sentence_idx if len(curr_sentence_idx) > 0 else [w2i['<unk>']])

                    if len(sentences_idx) > 1 and len(sentences_idx) < 1000:
                        data.append((sentences_idx[:self.params.max_sents_in_a_doc], tag))
                        new_adj.append(adj[idx])
                        new_fea.append(fea[idx])
                    else:
                        removed_idx.append(idx)
        print('removed_idx of {}: {}'.format(filename, len(removed_idx)))
        return data, new_adj, new_fea

    def read_testset_sentence_wise(self, filename, w2i, adj):
        df = pd.read_excel(filename)
        data, new_adj, new_fea = [], [], []
        count = 0
        adj, fea = zip(*adj)
        removed_idx = []
        for idx, row in tqdm(enumerate(df.values)):
            if self.params.DEBUG and idx >= DEBUG_NUM:
                break
            # sentences = row[2].split('.')
            sentences = sentence_tokenize(row[2])
            tag = int(row[0])
            # Tag id is reversed in this dataset
            tag = tag + 1 if tag == 0 else tag - 1
            sentences_idx = []
            for sentence in sentences:
                sentence = sentence.lower().replace("\n", " ").strip().split(" ")
                curr_sentence_idx = [w2i[x] for x in sentence]
                sentences_idx.append(curr_sentence_idx if len(curr_sentence_idx) > 0 else [w2i['<unk>']])
            if len(sentences_idx) > 1:
                data.append((sentences_idx, tag))
                new_adj.append(adj[count])
                new_fea.append(fea[count])
            else:
                removed_idx.append(idx)
            count += 1

        print('removed_idx of {}: {}'.format(filename, removed_idx))
        return data, new_adj, new_fea

def judge_data(fileName):
    key = fileName.split('.')[0]
    try:
        x = int(key)
        return True
    except:
        return False

def build_spr_coo(spr, device='cpu'):
    # {'indices': spr.indices(), 'value': spr.values(), 'size': spr.size()}
    if not isinstance(spr, dict):
        raise TypeError("Not recognized type of sparse matrix source: {}".format(type(spr)))
    tensor = torch.sparse.FloatTensor(spr['indices'], spr['value'], spr['size']).coalesce()
    return tensor if device == 'cpu' else tensor.to(device)

class DataSet(torch.utils.data.TensorDataset):
    def __init__(self, data, adj, fea, params, entity_description):
        super(DataSet, self).__init__()
        self.params = params
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.adjs = adj
        self.feas = fea
        self.entity_description = entity_description
        self.num_of_samples = len(self.sents)
        for i, a in enumerate(self.adjs):
            assert a[0].shape[0] == len(self.sents[i]),\
                "dim of adj does not match the num of sent, where the idx is {}".format(i)
            assert a[4].shape[0] == len(self.feas[i][1]), \
                "dim of adj does not match the num of entity, where the idx is {}".format(i)
            assert a[7].shape[0] == len(self.feas[i][2]), \
                "dim of adj does not match the num of topic, where the idx is {}".format(i)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx], self.adjs[idx], self.feas[idx]

    def collate(self, batch):
        sents, doc_lens_o, labels, adjs, feas = zip(*batch)
        # concatenate & padding
        doc_lens, curr_sents = [], []
        for doc in sents:
            doc_lens += [min(self.params.max_sent_len, len(x)) for x in doc]
            curr_sents += doc
        padded_sents = np.zeros((len(curr_sents), max(doc_lens)))
        for i, sen in enumerate(curr_sents):
            padded_sents[i, :len(sen)] = sen[:doc_lens[i]]
        documents = torch.from_numpy(padded_sents).long()

        new_feas, new_adjs = [], []
        fea_doc, fea_ent, fea_top = zip(*feas)
        for f in [fea_doc, fea_ent, fea_top]:
            fea = torch.from_numpy(np.array(sum([list(i.values()) for i in f], [])))
            new_feas.append(fea.long())
        for a in zip(*adjs):
            new_adjs.append(block_diag(a).float())

        labels = torch.from_numpy(np.array(labels)).long()
        sentPerDoc = torch.from_numpy(np.array([len(fea[0]) for fea in feas])).int()
        entiPerDoc = torch.from_numpy(np.array([len(fea[1]) for fea in feas])).int()
        topiPerDoc = torch.from_numpy(np.array([len(fea[2]) for fea in feas])).int()

        # concatenate & padding
        ent_lens, curr_sents = [], []
        for doc in fea_ent:
            doc = [self.entity_description[doc[idx]] for idx in range(len(doc))]
            ent_lens += [min(self.params.max_sent_len, len(x)) for x in doc]
            curr_sents += doc
        padded_sents = np.zeros((len(curr_sents), max(ent_lens)))
        for i, sen in enumerate(curr_sents):
            padded_sents[i, :len(sen)] = sen[:ent_lens[i]]
        ent_desc = torch.from_numpy(padded_sents).long()

        doc_lens = torch.from_numpy(np.array(doc_lens)).int()
        ent_lens = torch.from_numpy(np.array(ent_lens)).int()

        if self.params.node_type == 3:
            new_adjs = [new_adjs[0:3], new_adjs[3:6], new_adjs[6:9]]
            new_adjs[0][1].zero_()    # (√)text -> entity   (X)entity -> text
        elif self.params.node_type == 2:    # Document&Entiy
            new_adjs = [new_adjs[0:2], new_adjs[3:5]]
            new_feas = new_feas[0: 2]
            new_adjs[0][1].zero_()
        elif self.params.node_type == 1:    # Document&Topic
            new_adjs = [[new_adjs[0], new_adjs[2]], [new_adjs[6], new_adjs[8]]]
            new_feas = [new_feas[0], new_feas[2]]
            ent_desc, ent_lens, entiPerDoc = None, None, None
        elif self.params.node_type == 0:
            new_adjs = [[new_adjs[0]]]
            new_feas = [new_feas[0]]
            ent_desc, ent_lens, entiPerDoc = None, None, None
        else:
            raise Exception("Unknown node_type.")
        return documents, ent_desc, doc_lens, ent_lens, labels, new_adjs, new_feas, sentPerDoc, entiPerDoc


def block_diag(mat_list: list or tuple):
    shape_list = [m.shape for m in mat_list]
    bias = torch.LongTensor([0, 0])
    indices, values = [], []
    for m in mat_list:
        indices.append(m.indices() + bias.unsqueeze(1))
        values.append(m.values())
        bias += torch.LongTensor(list(m.shape))
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values, dim=0)
    res = torch.sparse.FloatTensor(indices, values, size=torch.Size(bias))
    return res

class freezable_defaultdict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        self.frozen = False
        self.default_factory = default_factory
        super(freezable_defaultdict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.frozen:
            return self.default_factory()
        else:
            self[key] = value = self.default_factory()
            return value

    def freeze(self):
        self.frozen = True
