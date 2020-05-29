import os, sys, json, torch
import argparse, datetime, time
import random, numpy as np
from util import Utils
from data_loader import DataLoader
from trainer import Trainer
from evaluator import Evaluator
from timeit import default_timer as timer
from print_log import Logger
from tqdm import tqdm

'''
node_type:
    '3 represents three types: Document&Entity&Topic; \n'
    '2 represents two types: Document&Entiy; \n'
    '1 represents two types: Document&Topic; \n'
    '0 represents only one type: Document. '
'''
# CUDA_VISIBLE_DEVICES_DICT = {0: '4',    1: '3',     2: '4',     3: '5'}
# MEMORY_DICT =               {0: 4000,   1: 9500,    2: 7600,    3: 8000}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for Fake News Detection')
    # Data Related
    # data_root_path = '/home/{}/GCN/FND/data/fakeNews/'.format(USERNAME)
    data_root_path = './data/fakeNews/'
    parser.add_argument("--root", type=str, default=data_root_path)
    parser.add_argument("--train", type=str, default=data_root_path + 'fulltrain.csv')
    parser.add_argument("--dev", type=str, default=data_root_path + 'balancedtest.csv')
    parser.add_argument("--test", type=str, default=data_root_path + 'test.xlsx',
                        help='Out of domain test set')
    parser.add_argument("--pte", type=str, default='', help='Pre-trained embeds')
    parser.add_argument("--entity_desc", type=str, help='entity description path.',
                        default=data_root_path + 'entityDescCorpus.pkl')
    parser.add_argument("--entity_tran", type=str, help='entity transE embedding path.',
                        default=data_root_path + 'entity_feature_transE.pkl')
    parser.add_argument("--adjs", type=str, default=data_root_path + 'adjs/')
    # Hyper-parameters
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--node_emb_dim", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--max_sent_len", type=int, default=50)
    parser.add_argument("--max_sents_in_a_doc", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--ntags", type=int, default=4)         # 4 or 2
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--pooling", type=str, default='max',
                        help='Pooling type: "max", "mean", "sum", "att". ')

    # parser.add_argument("--config", type=str, default='config_default',
    #                     help='Name for saving plots')
    parser.add_argument("--model_file", type=str, default='model_default.t7',
                        help='For evaluating a saved model')
    parser.add_argument("--plot", type=int, default=0, help='set to plot attn')
    parser.add_argument("--mode", type=int, default=0, help='0: train&test, 1:test')
    # parser.add_argument("--cuda", type=bool, default=True, help='use gpu to speed up or not')
    parser.add_argument("--cuda", type=bool, default=True, help='use gpu to speed up or not')
    parser.add_argument("--device", type=int, default=0, help='GPU ID. ')
    parser.add_argument("--HALF", type=bool, default=True, help='Use half tensor to save memory')

    parser.add_argument("--DEBUG", action='store_true', default=False, help='')
    parser.add_argument("--node_type", type=int, default=3,
                        help='3 represents three types: Document&Entity&Topic; \n'
                             '2 represents two types: Document&Entiy; \n'
                             '1 represents two types: Document&Topic; \n'
                             '0 represents only one type: Document. ')
    parser.add_argument('-r', "--repeat", type=int, default=1, help='')
    parser.add_argument('-s', "--seed", type=list, default=[5], help='')

    for dir in ["models/", "ckpt/", "plots/", "result/", "log/"]:
        if not os.path.exists(dir):   os.makedirs(dir)
    args = parser.parse_args()

    TIMENOW = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%m%d_%H%M")
    NODETYPE = {0: "D", 1: "DT", 2: "DE", 3: "DET"}[args.node_type]
    if args.mode == 0:
        MODELNAME = 'CompareNet_{}_{}_{}'.format(args.pooling.capitalize(), NODETYPE, TIMENOW)
        args.model_file = 'model_{}.t7'.format(MODELNAME)
        args.config = MODELNAME
        sys.stdout = Logger("./log/{}_{}.log".format(MODELNAME, TIMENOW))
    else:
        MODELNAME = args.model_file.split(".")[0]
        args.config = MODELNAME
        sys.stdout = Logger("./log/{}_{}.log".format(MODELNAME, TIMENOW))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.repeat = len(args.seed) if isinstance(args.seed, list) else args.repeat
    print("TimeStamp: {}\n".format(TIMENOW), json.dumps(vars(args), indent=2))
    return args



def main(params = None):
    if params is None:
        params = parse_arguments()
    SEED = params.seed
    t0 = time.time()
    s_t = timer()
    dl = DataLoader(params)

    u = Utils(params, dl)
    timeDelta = int(time.time()-t0)
    print("PreCost:", datetime.timedelta(seconds=timeDelta))
    for repeat in range(params.repeat):
        print("\n\n\n{0} Repeat: {1} {0}".format('-'*27, repeat))
        set_seed( SEED[repeat] if isinstance(SEED, list) else SEED )
        print("\n\n\n{0}  Seed: {1}  {0}".format('-'*27, SEED[repeat]))
        if params.mode == 0:
            # Start training
            trainer = Trainer(params, u)
            trainer.log_time['data_loading'] = timer() - s_t
            trainer.train()
            print(trainer.log_time)
            print("Total time taken (in seconds): {}".format(timer() - s_t))

            evaluator = Evaluator(params, u, dl)
            evaluator.evaluate()
        elif params.mode == 1:
            # Evaluate on the test set
            evaluator = Evaluator(params, u, dl)
            evaluator.evaluate()
        else:
            raise NotImplementedError("Unknown mode: {}".format(params.mode))


def set_seed(seed=9699):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':

    params = parse_arguments()
    set_seed(0)
    main(params)

