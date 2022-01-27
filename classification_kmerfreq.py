import logging
logging.basicConfig(filename='logs.log', 
                    format='%(asctime)s %(message)s',
                    level=logging.DEBUG,
                    filemode="w")
logging.info("clustering_kmerfreq.py")

import json 
import joblib # to load rff models
import numpy as np 
import pandas as pd

from collections import namedtuple
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec # to load doc2vec model
from sklearn.metrics import precision_recall_fscore_support
from src.spike_encoding import (
    Spike2KmerFreq,
)
from src.utils import spike_from_fasta
from parameters import PARAMETERS

# Classification models to try
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    "svm": SVC(),
    "random-forest": RandomForestClassifier(max_depth=2, random_state=0)
}

PATH_DATA = Path(PARAMETERS["PATH_DATA"])
PATH_SEQS = Path(PATH_DATA.joinpath("undersamples_fasta"))
PATH_MODELS = Path(PARAMETERS["PATH_MODELS"])
PATH_RESULTS = Path(PARAMETERS["PATH_RESULTS"])
PATH_RESULTS.mkdir(exist_ok=True)
PANGO_LINEAGES = PARAMETERS["PANGO_LINEAGES"]

# Labels 
# sklearn models require numbers as labels
logging.info("Loading labels")
pango2integer = {pango: j for j,pango in enumerate(PANGO_LINEAGES)}
integer2pango = {j: pango for pango,j in pango2integer.items()}

undersample_by_pango = pd.read_csv(PATH_DATA.joinpath("undersample_by_pango.csv"))
labels = {record.get("accessionID"): record.get("pango_lineage") for record in undersample_by_pango.to_dict("records")}

# Instantiate classifier
classifier = "random-forest"
clf = classifiers.get(classifier)

# instantiate encoder
logging.info("Loading encoder")
encoder = Spike2KmerFreq(k=3, # k for the k-mers to use
                         alphabet=None # by default, aminoacids are considered, 'X' included (21 in total)
                         )

# load model to preprocess input
logging.info("Loading model to generate features")
with open(str(PATH_MODELS.joinpath("rff_kmerfreq.sav")),"rb") as fp: 
    model_features = joblib.load(fp)

# Load data
logging.info("Loading datasets")
with open(PATH_DATA.joinpath("datasets.json")) as fp: 
    datasets = json.load(fp)

# Generate input vectors for RBFSampler (encoding)
logging.info("Generating encodings for train set")          
paths_train = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["train"]]

train_encoding = tuple(
                    encoder(spike_from_fasta(path)) for path in paths_train
                    )
X_train = np.vstack(train_encoding)

# get low dimensional features
X_train = model_features.transform(X_train)

# labels train
logging.info("Labels for train set")
labels_train = [labels[accessionID] for accessionID in datasets["train"]]
y_train = [pango2integer[pango] for pango in labels_train]

logging.info("Generating encodings for test set")    
paths_test  = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["test"]]
test_encoding = tuple(
                    encoder(spike_from_fasta(path)) for path in paths_test
                    )
X_test = np.vstack(test_encoding)

# get low dimensional features
X_test = model_features.transform(X_test)

# labels test
logging.info("Labels for test set")
labels_test = [labels[accessionID] for accessionID in datasets["test"]]
y_test = [pango2integer[pango] for pango in labels_test]

for classifier, clf in classifiers.items():
    logging.info(f"Train classifier {classifier}")
    # train
    clf.fit(X_train,y_train)
    logging.info("Training done")

    # Save model
    logging.info("Save trained model")
    joblib.dump(clf, str(PATH_MODELS.joinpath(f"{classifier}-kmerfreq.sav")))

    # Test
    logging.info("Test classifier")
    preds_test = clf.predict(X_test)

    logging.info("Metrics")
    y_true = labels_test
    y_pred = [integer2pango[integer] for integer in preds_test]
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=PANGO_LINEAGES)

    list_metrics = []
    Metrics = namedtuple("Metrics", ["pango_lineage","precision", "recall", "fscore", "support"])
    for j,pango in enumerate(PANGO_LINEAGES): 
        list_metrics.append(
            Metrics(pango, precision[j], recall[j], fscore[j], support[j])
        )

    pd.DataFrame(list_metrics).to_csv(PATH_RESULTS.joinpath(f"test-{classifier}-kmerfreq.csv"))