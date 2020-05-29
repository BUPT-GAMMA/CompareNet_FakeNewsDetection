from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
from models import Classifier
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Utils:
    def __init__(self, params, dl):
        self.params = params
        self.data_loader = dl
        self.HALF = params.HALF

    @staticmethod
    def to_half(arr):
        if arr is None:
            return arr
        if isinstance(arr, list) or isinstance(arr, tuple):
            return [Utils.to_half(a) for a in arr]
        elif isinstance(arr, torch.FloatTensor) or isinstance(arr, torch.sparse.FloatTensor):
            return arr.half()
        else:
            return arr

    def to_gpu(self, arr, cuda):
        if self.params.HALF:
            arr = Utils.to_half(arr)

        if not cuda or arr is None:
            return arr
        if isinstance(arr, list) or isinstance(arr, tuple):
            return [Utils.to_gpu(a, cuda) for a in arr]
        else:
            try:
                return arr.cuda()
            except:
                return arr
        # elif isinstance(arr[0], int) or isinstance(arr[0], float):
        #     return arr
        # else:
        #     raise TypeError("Unknown type of input of 'utils.py/Utils/to_gpu': {}.".format(type(arr)))

    def get_dev_loss_and_acc(self, model, loss_fn):
        losses = []; hits = 0; total = 0; outOfMemoryCnt = 0
        model.eval()
        for inputs in tqdm(self.data_loader.dev_data_loader):
            # torch.cuda.empty_cache()
            with torch.no_grad():
                try:
                    documents, ent_desc, doc_lens, ent_lens, y_batch, adj_lists, feature_lists, sentPerDoc, entiPerDoc = \
                        [self.to_gpu(i, self.params.cuda and torch.cuda.is_available()) for i in inputs]
                    logits = model(documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc)
                    loss = loss_fn(logits, y_batch)
                    hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
                    total += sentPerDoc.shape[0]
                    losses.append(loss.item())
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        # outOfMemory += 1
                        continue
                    else:
                        print(e)
                        exit()
                except Exception as e:
                    print(e)
                    exit()
        if outOfMemoryCnt > 0:
            print("outOfMemoryCnt when validating: ", outOfMemoryCnt)
        return np.asscalar(np.mean(losses)), hits / total

    def train(self, save_plots_as, pretrained_emb=None):
        model: nn.Module = Classifier(self.params, vocab_size=len(self.data_loader.w2i), pte=pretrained_emb)
        if self.params.HALF:
            model.half()
        loss_fn = torch.nn.CrossEntropyLoss()
        if self.params.cuda:
            model = model.cuda()
        # optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, eps=1e-4)

        # Variables for plotting
        train_losses, dev_losses, train_accs, dev_accs = [], [], [], []
        s_t = timer()
        prev_best = 0
        patience = 0
        outOfMemory = 0
        # Start the training loop
        for epoch in range(1, self.params.max_epochs + 1):
            model.train()
            train_loss, hits, total = 0, 0, 0
            # train_data_loader = self.data_loader.train_data_loader \
            #                     if not self.params.DEBUG else self.data_loader.test2_data_loader
            for inputs in tqdm(self.data_loader.train_data_loader):
                # torch.cuda.empty_cache()
                try:
                    documents, ent_desc, doc_lens, ent_lens, y_batch, adj_lists, feature_lists, sentPerDoc, entiPerDoc = \
                                                        [self.to_gpu(i, self.params.cuda and torch.cuda.is_available()) for i in inputs]
                    total += sentPerDoc.shape[0]
                    logits = model(documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc)
                    if torch.isnan(logits).any():
                        print('stop here')
                        # model(documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc)
                    loss = loss_fn(logits, y_batch)
                    # Book keeping
                    train_loss += loss.item()
                    hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
                    # Back-prop
                    optimizer.zero_grad()  # Reset the gradients
                    loss.backward()  # Back propagate the gradients
                    optimizer.step()  # Update the network

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        outOfMemory += 1
                        continue
                    else:
                        print(e)
                        exit()
                except Exception as e:
                    print(e)
                    exit()
            print("Times of out of memory: ", outOfMemory)
            # Compute loss and acc for dev set
            dev_loss, dev_acc = self.get_dev_loss_and_acc(model, loss_fn)
            train_loss = train_loss / len(self.data_loader.train_data_loader)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            train_accs.append(hits / total)
            dev_accs.append(dev_acc)
            tqdm.write("Epoch: {}, Train loss: {:.4f}, Train acc: {:.4f}, Dev loss: {:.4f}, Dev acc: {:.4f}".format(
                        epoch, train_loss, hits / total, dev_loss, dev_acc))
            if dev_acc < prev_best:
                patience += 1
                if patience == 3:
                    # Learning rate annealing
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2
                    optimizer.load_state_dict(optim_state)
                    tqdm.write('Dev accuracy did not increase, reducing the learning rate by 2!!!')
                    patience = 0
            else:
                prev_best = dev_acc
                # Save the model
                torch.save(model.state_dict(), "ckpt/model_{}.t7".format(save_plots_as))

        # Acc vs time plot
        fig = plt.figure()
        plt.plot(range(1, self.params.max_epochs + 1), train_accs, color='b', label='train')
        plt.plot(range(1, self.params.max_epochs + 1), dev_accs, color='r', label='dev')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend()
        plt.xticks(np.arange(1, self.params.max_epochs + 1, step=4))
        fig.savefig('result/' + '{}_accuracy.png'.format(save_plots_as))

        return timer() - s_t
