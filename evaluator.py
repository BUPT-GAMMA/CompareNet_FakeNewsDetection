import torch, json
from models import Classifier
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
# from util import Utils

class Evaluator:
    def __init__(self, params, utils, data_loader):
        self.params = params
        self.utils = utils
        self.data_loader = data_loader

    def get_sentences_from_indices(self, docs):
        actual_sentences = []
        for doc, sent_lens in docs:
            sentences = []
            for i, sent in enumerate(doc):
                sentences.append(' '.join([self.data_loader.i2w[int(wid)] for wid in sent[:sent_lens[i]]]))
            actual_sentences.append(sentences)
        return actual_sentences

    def _evaluate_aux(self, model, data_loader):
        hits = 0
        total = 0
        all_actual = None
        all_predicted = None
        for inputs in tqdm(data_loader):
            with torch.no_grad():
                try:
                    documents, ent_desc, doc_lens, ent_lens, y_batch, adj_lists, feature_lists, sentPerDoc, entiPerDoc = \
                        [self.utils.to_gpu(i, self.params.cuda and torch.cuda.is_available()) for i in inputs]
                    total += sentPerDoc.shape[0]
                    logits = model(documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc)
                    predicted = torch.argmax(logits, dim=1)
                    hits += torch.sum(predicted == y_batch).item()
                    all_predicted = predicted.cpu().data.numpy() if all_predicted is None \
                        else np.concatenate((all_predicted, predicted.cpu().data.numpy()))
                    labels = y_batch.cpu().numpy()
                    all_actual = labels if all_actual is None else np.concatenate((all_actual, labels))
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
        accuracy = hits / total
        return accuracy, all_actual, all_predicted

    def evaluate(self):
        print(json.dumps(vars(self.params), indent=2))

        model: torch.nn.Module = Classifier(self.params, vocab_size=len(self.data_loader.w2i), pte=None)
        if self.utils.HALF:
            model.half()
        if torch.cuda.is_available():
            model = model.cuda()
        # Load the model weights
        model.load_state_dict(torch.load("ckpt/" + self.params.model_file, map_location=lambda storage, loc: storage))

        model.eval()

        # This dataset is only available for the binary classifier
        if self.params.ntags == 2:
            accuracy, all_actual, all_predicted = self._evaluate_aux(model, self.data_loader.test_data_loader)
            prec_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(all_actual, all_predicted, average='macro')
            prec_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(all_actual, all_predicted, average='micro')
            print("Accuracy on the OOD test set 1: {:.4f}".format(accuracy))
            print("Precision on the OOD test set 1 macro / micro: {:.4f}, {:.4f}".format(prec_mac, prec_mic))
            print("Recall on the OOD test set 1 macro / micro: {:.4f}, {:.4f}".format(recall_mac, recall_mic))
            print("F1 on the OOD test set 1 macro / micro: {:.4f}, {:.4f}".format(f1_mac, f1_mic))
            print("Latex: {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f}".format(accuracy*100, prec_mac*100, recall_mac*100, f1_mac*100))
            print("----------------------------------------------------------------------")

        accuracy, all_actual, all_predicted = self._evaluate_aux(model, self.data_loader.test_data_loader_2)
        prec_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(all_actual, all_predicted, average='macro')
        prec_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(all_actual, all_predicted, average='micro')
        print("Accuracy on the OOD test set 2: {:.4f}".format(accuracy))
        print("Precision on the OOD test set 2 macro / micro: {:.4f}, {:.4f}".format(prec_mac, prec_mic))
        print("Recall on the OOD test set 2 macro / micro: {:.4f}, {:.4f}".format(recall_mac, recall_mic))
        print("F1 on the OOD test set 2 macro / micro: {:.4f}, {:.4f}".format(f1_mac, f1_mic))
        print("Latex: {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f}".format(accuracy * 100, prec_mac * 100, recall_mac * 100, f1_mac * 100))

        if self.params.ntags == 4:
            results = confusion_matrix(all_actual, all_predicted)
            df_cm = pd.DataFrame(results, index=[i for i in ["Satire", "Hoax", "Propaganda", "Trusted"]],
                                 columns=[i for i in ["Satire", "Hoax", "Propaganda", "Trusted"]])
            sns_plot = sn.heatmap(df_cm, annot=True, fmt='g')
            plt.yticks(rotation=45)
            sns_plot.get_figure().savefig('plots/cm.png')

        print("----------------------------------------------------------------------")
        accuracy, all_actual, all_predicted = self._evaluate_aux(model, self.data_loader.dev_data_loader)
        prec_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(all_actual, all_predicted, average='macro')
        prec_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(all_actual, all_predicted, average='micro')
        print("Accuracy on the dev set: {:.4f}".format(accuracy))
        print("Precision on the dev set macro / micro: {:.4f}, {:.4f}".format(prec_mac, prec_mic))
        print("Recall on the dev macro / micro: {:.4f}, {:.4f}".format(recall_mac, recall_mic))
        print("F1 on the dev macro / micro: {:.4f}, {:.4f}".format(f1_mac, f1_mic))
        print("Latex: {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f}".format(accuracy * 100, prec_mac * 100, recall_mac * 100, f1_mac * 100))


