import torch
import torch.nn as nn

class NucPreprocess():
    def __init__(self, sequences):
        self.sequences = sequences
        self.classes = 5 #A,T,C,G,N
        self.encoding_dict = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4, 'Y':4, 'R':4, 'K':4, 'M':4, 'S':4, 'W':4, 'V':4, 'B':4, 'H':4, 'D':4}
        self.decoding_dict = {0:'A', 1:'T', 2:'C', 3:'G', 4:'N'}
        
        
    def onehot_for_nuc(self):
        x = list()
        for seq in self.sequences:
            precoding = [ self.encoding_dict[nuc.upper()] for nuc in seq ]
            classes = self.classes
            index = torch.unsqueeze(torch.tensor(precoding).long(),dim=1)
            src = torch.ones(len(seq), classes).long()
            
            onehot = torch.zeros(len(seq), classes).long()
            onehot.scatter_(dim=1, index=index, src=src)
            
            # Droup N base dim in coding
            onehot = onehot[:,:4]
            x.append(onehot.short())
            
        return x
        
        
    def decode_for_nuc(self, coded_seq):
        source_seq = ''
        for coding in coded_seq:
            if torch.sum(coding):
                new_nuc = self.decoding_dict[int(coding.nonzero().squeeze())]
                source_seq += new_nuc
            else:
                source_seq += 'N'
                
        return source_seq

class KmerPreprocess():
    def __init__(self, sequences, k=4):
        """
        Parameters:
        - sequences: list of DNA sequences.
        - k: the size of k-mer, default is 4.
        """
        self.sequences = sequences
        self.k = k
        self.classes = 4 ** k  # Total number of possible k-mers
        self.kmer_dict = self._build_kmer_dict()
        self.reverse_kmer_dict = {v: k for k, v in self.kmer_dict.items()}

    def _build_kmer_dict(self):
        """
        Create a dictionary mapping all possible k-mers to unique integer indices.
        """
        bases = ['A', 'T', 'C', 'G']
        kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        return {kmer: idx for idx, kmer in enumerate(kmers)}

    def kmer_encode(self):
        """
        Convert sequences into k-mer numeric representations.
        """
        encoded_kmers = []
        for seq in self.sequences:
            kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]  # Sliding window
            encoded = [self.kmer_dict.get(kmer, -1) for kmer in kmers]  # Encode k-mers
            encoded_kmers.append(torch.tensor(encoded).long())
        return encoded_kmers

    def kmer_onehot(self):
        """
        Convert k-mers into one-hot encoding representations.
        """
        onehot_kmers = []
        for seq in self.sequences:
            kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]  # Sliding window
            encoded = [self.kmer_dict.get(kmer, -1) for kmer in kmers]  # Encode k-mers
            # Create one-hot encoding
            index = torch.tensor(encoded).long().unsqueeze(1)  # Shape (num_kmers, 1)
            src = torch.ones(len(encoded), self.classes).long()  # Create source matrix
            onehot = torch.zeros(len(encoded), self.classes).long()  # Initialize one-hot matrix
            onehot.scatter_(dim=1, index=index, src=src)  # Assign 1s to appropriate indices
            onehot_kmers.append(onehot)
        return onehot_kmers
