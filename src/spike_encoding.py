"""
Encode a spike sequence into frequency k-mer vector
"""
import numpy as np
from typing import List
from itertools import product
from collections import defaultdict

from gensim.models.doc2vec import TaggedDocument # training data


AMINOACIDS = list("ACDEFGHIKLMNPQRSTVWXY")

class Spike2KmerFreq:
    "Compute the frequency vector of k-mers for the spike sequence"
    def __init__(self, k:int=3, alphabet: List[str] = None) -> np.ndarray:
        self.k = k
        self.alphabet = AMINOACIDS if alphabet is None else alphabet
        self.kmers2pos = {"".join(tuples): pos for pos,tuples in enumerate(product(self.alphabet,repeat=self.k))}

    def __call__(self, seq: str): 
        "Return a vector with frequency of k-mers"
        # generate all k-mers in the sequence
        last_j = len(seq) - self.k + 1   
        kmers_seq  = (seq[i:(i+self.k)] for i in range(last_j))

        # count k-mers
        count_kmers = defaultdict(int)
        for kmer in kmers_seq: 
            count_kmers[kmer] +=1

        # build frequency with kmers vector (lexicographiclly)
        freq_kmers = np.zeros(len(self.kmers2pos))
        for kmer, freq in count_kmers.items():
            freq_kmers[self.kmers2pos[kmer]] = freq

        return freq_kmers
        
class Spike2OneHotEncoding:
    """Compute a one hot encode VECTOR for each spike sequence
    The vector will be flatten, i.e. first 21 position correspond to the first aminoacid, and so on
    """
    def __init__(self, alphabet: List[str] = None, len_seqs = 1273):
        self.alphabet = AMINOACIDS if alphabet is None else alphabet
        self.len_seqs = len_seqs
        self.aminoacid2pos = {aminoacid: pos for pos,aminoacid in enumerate(self.alphabet)}

    def __call__(self, seq: str): 
        "Sequence to one-hot encoding based on aminoacid"
        array = np.zeros((self.len_seqs ,len(self.alphabet)))
        for row, aminoacid in enumerate(seq): 
            col = self.aminoacid2pos[aminoacid]
            array[row,col] = 1

        return array.flatten()

class Doc2VecEncoding:
    """Obtain the encoding needed for a Doc2Vec model using gensim library
    For test and inference, it is a list of the tokens (k-mers)
    For training, it needs an extra attribute, a tag
    """

    def __init__(self, k: int = 3, train_mode: bool = False):
        self.k = k
        self.train_mode = train_mode # train_model=True requires a tag
        self.tag = 0 if train_mode is True else None

    def __call__(self, seq: str):
        # extract k-mers as list
        if self.train_mode is True: 
            encode = self.encode_train(seq)
        else:
            encode = self.encode_test(seq)
        return encode
    
    def encode_train(self, seq: str):
        "Return a tagged list of k-mers"
        list_kmers = self.seq2kmers(seq)
        encode = TaggedDocument(list_kmers, [self.tag])
        
        # update tag for the next sequence
        self.tag +=1 
        return encode
    
    def encode_test(self, seq: str):
        "Return a list of k-mers"
        return self.seq2kmers(seq)
    
    def seq2kmers(self, seq: str):
        "Return a list with k-mers from the sequence"
        last_j = len(seq) - self.k + 1 
        kmers_seq  = (seq[i:(i+self.k)] for i in range(last_j)) 
        return list(kmers_seq)