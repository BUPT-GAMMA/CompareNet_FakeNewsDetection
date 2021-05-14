Code for the ACL 2021 paper "Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge"


Make sure the following files are present as per the directory structure before running the code,
```
FakeNewsDetection
├── README.md
├── *.py
└───models
|   └── *.py 
└───data
    ├── fakeNews
    │   ├── adjs
    │   │   ├── train
    │   │   ├── dev
    │   │   └── test
    │   ├── fulltrain.csv
    │   ├── balancedtest.csv
    │   ├── test.xlsx
    │   ├── entityDescCorpus.pkl
    │   └── entity_feature_transE.pkl
    └── stopwords_en.txt

```

balancedtest.csv and fulltrain.csv can be obtained from https://homes.cs.washington.edu/~hrashkin/fact_checking_files/newsfiles.tar.gz

test.xsls is basically the SLN dataset according to the paper. You can obtain this dataset from http://victoriarubin.fims.uwo.ca/news-verification/data-to-go/



# Dependencies

Our code runs on the GeForce RTX 2080 Ti (11GB), with the following packages installed:
```
python 3.7
torch 1.3.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
```



# Run

Train and test,
```
python main.py --mode 0
```

Test,
```
python main.py --mode 1 --model_file MODELNAME
```

# Citation
If you make advantage of our model in your research, please cite the following in your manuscript:
```
@inproceedings{linmei2019heterogeneous,
 title={Heterogeneous graph attention networks for semi-supervised short text classification},
 author={Linmei, Hu and Yang, Tianchi and Shi, Chuan and Ji, Houye and Li, Xiaoli},
 booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
 pages={4823--4832},
 year={2019}
}
```

