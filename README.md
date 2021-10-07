Code for the ACL 2021 paper "[Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge](https://aclanthology.org/2021.acl-long.62/)"


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

balancedtest.csv and fulltrain.csv can be obtained from https://drive.google.com/file/d/1njY42YQD5Mzsx2MKkI_DdtCk5OUKgaqq/view?usp=sharing (thanks to https://github.com/MysteryVaibhav/fake_news_semantics)

test.xsls is basically the SLN dataset according to the paper. You can obtain this dataset from http://victoriarubin.fims.uwo.ca/news-verification/data-to-go/

UPDATE: The datasets from the above links seem to be changed. I have found the data in my local machine and uploaded it to the release(raw_data.zip): https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset

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
@inproceedings{linmei2021compare,
    title = "Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge",
    author = "Hu, Linmei  and  Yang, Tianchi  and  Zhang, Luhao  and  Zhong, Wanjun  and  Tang, Duyu  and  Shi, Chuan  and  Duan, Nan  and  Zhou, Ming",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.62",
    doi = "10.18653/v1/2021.acl-long.62",
    pages = "754--763",
}

```

