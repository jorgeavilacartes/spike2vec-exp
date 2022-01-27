"""Extract undersampled sequences to individual files
"""
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from parameters import PARAMETERS
PATH_FASTA = Path(PARAMETERS["PATH_FASTA"])
PATH_DATA = Path(PARAMETERS["PATH_DATA"])
PATH_SAVE_SEQS = PATH_DATA.joinpath("undersamples_fasta")
PATH_SAVE_SEQS.mkdir(exist_ok=True)

fasta_order = pd.read_csv(PATH_DATA.joinpath("fasta_order.csv"))
undersample_by_pango = pd.read_csv(PATH_DATA.joinpath("undersample_by_pango.csv"))
order_selected_seqs  = undersample_by_pango.merge(fasta_order, on="accessionID", how="left")["pos_fasta"].tolist()
order_selected_seqs.sort()
set_accession_id = set(order_selected_seqs)

# Read fasta with all sequences from GISAID
with open(PATH_FASTA) as handle:
    for pos_fasta, record in enumerate(SeqIO.parse(handle, "fasta")):
        
        # save sequence if it was selected
        if pos_fasta in set_accession_id:
            # save sequence in a fasta file "<accession_id>.fasta" 
            filename = record.id.split("|")[3] # accession id
            path_save = PATH_SAVE_SEQS.joinpath(f"{filename}.fasta")
            if not path_save.is_file():
                SeqIO.write(record, path_save, "fasta") 
            # remove from the set to be saved   
            set_accession_id.remove(pos_fasta)
        
        # if all sequences has been saved, break the loop
        if not set_accession_id:
            break