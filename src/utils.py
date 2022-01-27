from Bio import SeqIO

def spike_from_fasta(path):
    record = next(SeqIO.parse(path, "fasta"))
    seq = record.seq
    seq = seq.replace("*","")
    return seq

def padding(seq, length=1273):
    if len(seq)>length:
        return seq[:length]
    else: 
        return seq + "X"*(length-len(seq))