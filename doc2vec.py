"""
Train a doc2vec model for spike sequences
"""
import json
from tqdm import tqdm
from pathlib import Path
from gensim.models.doc2vec import (
    Doc2Vec,
    TaggedDocument, # training data
)
from src.spike_encoding import Doc2VecEncoding
from src.utils import spike_from_fasta
from parameters import PARAMETERS

PATH_DATA = Path(PARAMETERS["PATH_DATA"])
RANDOM_STATE = PARAMETERS["RANDOM_STATE"]
RFF_SIZE = PARAMETERS["RFF_SIZE"]
PATH_SEQS = PATH_DATA.joinpath("undersamples_fasta")
DOC2VEC_SIZE = PARAMETERS["DOC2VEC_SIZE"]
DOC2VEC_EPOCHS = PARAMETERS["DOC2VEC_EPOCHS"]
PATH_MODELS = Path(PARAMETERS["PATH_MODELS"])
PATH_MODELS.mkdir(exist_ok=True)

# -1- Instantiate encoder for training data
encoder = Doc2VecEncoding(
            k=3, 
            train_mode=True
            )
# -2- load data
with open(PATH_DATA.joinpath("datasets.json")) as fp: 
    datasets = json.load(fp)

paths_train = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["train"]]
paths_test  = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["test"]]

train_corpus = tuple(
                    encoder(spike_from_fasta(path)) for path in paths_train
                    )
# -3- Train model
# define model
model = Doc2Vec(vector_size=DOC2VEC_SIZE, min_count=1, epochs=DOC2VEC_EPOCHS)
model.build_vocab(train_corpus)

# train model
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# save model
model.save(str(PATH_MODELS.joinpath("spike_doc2vec.model")))