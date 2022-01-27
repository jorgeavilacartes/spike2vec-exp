import random 
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
from Bio import SeqIO
import pandas as pd

from parameters import PARAMETERS

COLS_METADATA = ["Accession ID", "Pango lineage","Is low coverage?","Host"]
PATH_METADATA = Path(PARAMETERS["PATH_METADATA"])
PATH_FASTA =  Path(PARAMETERS["PATH_FASTA"]) 
PANGO_LINEAGES = PARAMETERS["PANGO_LINEAGES"]
SAMPLES_PER_PANGO = 10_000
RANDOM_STATE = PARAMETERS["RANDOM_STATE"]
random.seed(RANDOM_STATE)

# Extract accession ID from FASTA
FastaOrder = namedtuple("FastaOrder", ["pos_fasta","accessionID"])
accessionID_fasta = []

with open(PATH_FASTA) as fp: 
    pos_fasta = 0
    for record in SeqIO.parse(fp, "fasta"):
        fasta_id = record.id
        try: 
            accessionID = fasta_id.split("|")[3]
        except:
            accessionID = None
        accessionID_fasta.append(FastaOrder(pos_fasta,accessionID))
        pos_fasta +=1

        if pos_fasta >1_000_000: break

pd.DataFrame(accessionID_fasta).to_csv("data/fasta_order.csv",index=False) 

# Load and filter metadata
metadata = pd.read_csv(PATH_METADATA, sep="\t", usecols=COLS_METADATA, 
                        nrows=100_000)
# Remove NaN in Clades and not-complete sequences
metadata.dropna(axis="rows",
            how="any",
            subset=["Pango lineage", "Is low coverage?"], 
            inplace=True,
            )
available_data=tuple(fasta_order.accessionID 
                        for fasta_order in accessionID_fasta
                        if fasta_order.accessionID is not None)
                        
metadata.query("`Accession ID` in {} and `Host`=='Human'".format(available_data), inplace=True)

# subsample 
SamplePango = namedtuple("SamplePango", ["accessionID","pango_lineage"])
list_fasta_selected = []
for pango in tqdm(PANGO_LINEAGES):
    samples_pango = metadata.query(f"`Pango lineage` == '{pango}'")["Accession ID"].tolist()
    random.shuffle(samples_pango)
    # select 'SAMPLES_PER_CLADE' samples for each clade, or all of them if available samples are less than required
    list_fasta_selected.extend([SamplePango(accessionID, pango) for accessionID in samples_pango[:SAMPLES_PER_PANGO]])

pd.DataFrame(list_fasta_selected).to_csv("data/undersample_by_pango.csv")