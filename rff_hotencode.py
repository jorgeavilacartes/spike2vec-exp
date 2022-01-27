"""
Random Fourier Features of spike sequences
    z: R^d -> R^D
Use frequency vector of k-mers (k=3)
"""
import json
import joblib
import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.kernel_approximation import RBFSampler
from parameters import PARAMETERS
from src.spike_encoding import Spike2OneHotEncoding
from src.utils import spike_from_fasta, padding

PATH_DATA = Path(PARAMETERS["PATH_DATA"])
RANDOM_STATE = PARAMETERS["RANDOM_STATE"]
RFF_SIZE = PARAMETERS["RFF_SIZE"]
PATH_SEQS = PATH_DATA.joinpath("undersamples_fasta")
PATH_MODELS = Path(PARAMETERS["PATH_MODELS"])
PATH_MODELS.mkdir(exist_ok=True)
# Instantiate random fourier features
rbf_features = RBFSampler(gamma=1, n_components=RFF_SIZE, random_state = RANDOM_STATE)

# instantiate encoder
encoder = Spike2OneHotEncoding()

# load data
with open(PATH_DATA.joinpath("datasets.json")) as fp: 
    datasets = json.load(fp)

# -1- Generate input vectors for RBFSampler (encoding)          
paths_train = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["train"]]
paths_test  = [PATH_SEQS.joinpath(f"{accessionID}.fasta") for accessionID in datasets["test"]]

train_encoding = tuple(
                    encoder(padding(spike_from_fasta(path))) for path in paths_train
                    )
X_train = np.vstack(train_encoding)

# -2- Fit model
rbf_features.fit(X_train)

# -3- Save model
joblib.dump(rbf_features, str(PATH_MODELS.joinpath("rff_hotencode.sav")))