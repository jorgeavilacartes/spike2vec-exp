"""
Decide if I will work with GISAID data directly or with the preprocessed data from the paper
Split undersampled GISAID metadata into train and test sets
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from parameters import PARAMETERS

PATH_DATA = Path(PARAMETERS["PATH_DATA"])
TRAIN_SIZE = PARAMETERS["TRAIN_SIZE"]
RANDOM_STATE = PARAMETERS.get("RANDOM_STATE")

undersample=pd.read_csv("data/undersample_by_pango.csv")

X = undersample["accessionID"].tolist()*2
y = undersample["pango_lineage"].tolist()*2

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                  train_size=TRAIN_SIZE,
                                                  random_state=RANDOM_STATE, # reproducibility 
                                                  stratify=y # balanced based on pango_lineages
                                                 )

# -3- Save splitted dataset
with open(str(PATH_DATA.joinpath("datasets.json")),"w") as fp: 
    json.dump({"train": X_train, "test": X_test}, fp, indent=2)

# generate a summary of labels by set
summary_labels = pd.DataFrame.from_dict(
    {"train": dict(Counter(y_train)), "test": dict(Counter(y_test))}
)

summary_labels["TOTAL"] = summary_labels.sum(axis=1) # compute total by label
summary_labels.to_csv(PATH_DATA.joinpath("summary_labels.csv")) # save summary