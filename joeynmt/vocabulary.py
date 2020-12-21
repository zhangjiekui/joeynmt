# coding: utf-8

"""
Vocabulary module
"""
from collections import defaultdict, Counter
from typing import List
import numpy as np

from torchtext.data import Dataset

from joeynmt.constants import UNK_TOKEN, DEFAULT_UNK_ID, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN

class Vocabulary:
    '''Vocabulary represents mapping between tokens and indices. Vocabulary代表着tokens（标记）和indices（索引）之间的映射'''
    def __init__(self, tokens:List[str]=None, file:str=None, encoding='utf-8',lower=False) ->None:
        '''
        Create vocabulary from list of tokens or file.从标记列表或文件来创建词表
        Special tokens are added if not already in file or list.加入特殊标记，即使这些特殊标记没有出现在标记列表或文件中
        File format: token with index i is in line i. 文件格式：索引为i的标记，对应文件中的第i行
        :param tokens: list of tokens 标记列表
        :param file: file to load vocabulary from 文件路径，将从该文件中载入词表
        '''
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        
        # special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.stoi = defaultdict(DEFAULT_UNK_ID)
        self.itos = []
        if tokens is not None:
            self._from_list(tokens)
        if file is not None:
            self._from_file(file,encoding = encoding,lower = lower)
    def add_tokens(self, tokens:List[str],lower=False) ->None:
        """
        Add list of tokens to vocabulary
        :param tokens: list of tokens to add to the vocabulary
        """
        for token in tokens:
            new_index=len(self.itos)
            if token not in self.itos:
                token=token.strip()
                if lower:
                    token=token.lower()
                self.itos.append(token)
                self.stoi[token]=new_index
        
    def _from_list(self, tokens: List[str] = None,lower=False) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.
        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials+tokens,lower=lower)
        assert len(self.stoi) == len(self.itos)
        
    def _from_file(self, file: str,encoding='utf-8',lower=False) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.
        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r",encoding=encoding) as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens,lower=lower)
    
    def __str__(self) -> str:
        return self.stoi.__str__()
    
    def to_file(self, file: str,encoding='utf-8') -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.
        :param file: path to file where the vocabulary is written
        """
        with open(file, "w",encoding=encoding) as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))
    
    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary
        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)
    
    def _array_to_sentence(self, array: np.array, cut_at_eos=True, skip_pad=True) -> List[str]:
        """
        1D array --> sentence
        
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.
        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
        return sentence
    
    def _arrays_to_sentences(self, arrays: np.array, cut_at_eos=True,skip_pad=True) -> List[List[str]]:
        """
        2D array --> sentences
        
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.
        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self._array_to_sentence(array=array, cut_at_eos=cut_at_eos,skip_pad=skip_pad))
        return sentences
    
    def array_to_sentence(self, array: np.array, cut_at_eos=True,skip_pad=True) -> List[List[str]]:
        dim=len(array.shape)
        if dim==1:
            return self._array_to_sentence(array, cut_at_eos, skip_pad)
        if dim==2:
            return self._arrays_to_sentences(array, cut_at_eos, skip_pad)
        else:
            raise ValueError(f"要求参数array必须是1D或2D的，但现在输入的参数array是{dim}D的！")
    def arrays_to_sentences(self, array: np.array, cut_at_eos=True,skip_pad=True) -> List[List[str]]:
        return self.array_to_sentence(array,cut_at_eos,skip_pad)
            
def build_vocab(field: str, max_size: int, min_freq: int=0, dataset: Dataset = None , vocab_file: str = None, encoding='utf-8',lower=False) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.
    :param field: attribute e.g. "src"
    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param dataset: dataset to load data for field from
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `dataset` or `vocab_file`
    """

    if vocab_file is not None:
        # load it from file
        vocab = Vocabulary(file=vocab_file, encoding=encoding, lower=lower)
    elif dataset is not None:
        # create newly
        def filter_min(counter: Counter, min_freq: int):
            """ Filter counter by min frequency """
            filtered_counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
            return filtered_counter

        def sort_and_cut(counter: Counter, limit: int):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
            return vocab_tokens

        tokens = []
        for i in dataset.examples:
            if field == "src":
                tokens.extend(i.src)
            elif field == "trg":
                tokens.extend(i.trg)

        counter = Counter(tokens)
        if min_freq > 0:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        vocab = Vocabulary(tokens=vocab_tokens)
        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[DEFAULT_UNK_ID()] == UNK_TOKEN

    else:
        raise ValueError(f"要求参数dataset或者vocab_file 至少有一个不为空！")

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab  
        
