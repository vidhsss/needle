import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            uid = len(self.idx2word)
            self.word2idx[word] = uid
            self.idx2word.append(word)
        else:
            uid = self.word2idx[word]
        return uid
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'ptb.train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'ptb.test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        # ids = []
        # # eos_id = self.dictionary.add_word("<eos>")

        # def tokenize_one_line(line):
        #     words = line.split()
        #     for word in words:
        #         ids.append(self.dictionary.add_word(word))
        #     ids.append(self.dictionary.add_word("<eos>"))

        # with open(path, "r") as f:
        #     if max_lines:
        #         for _ in range(max_lines):
        #             tokenize_one_line(f.readline())
        #     else:
        #         for line in f:
        #             tokenize_one_line(line)
        # return ids
        ids = []
        # eos_id = self.dictionary.add_word("<eos>")
        
        with open(path, "r") as file:
            lines = file.readlines()[:max_lines] if max_lines else file
            for line in lines:
                words = line.strip().split()
                ids.extend(self.dictionary.add_word(word) for word in words)
                ids.append(self.dictionary.add_word("<eos>"))  # Append end-of-sentence token ID for each line
        
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    # size = len(data) // batch_size
    # data = np.array(data[:size * batch_size]).reshape((size, batch_size))
    # return data
    nbatch = len(data) // batch_size
    
    # Trim data to fit into batches evenly
    trimmed_data = data[:nbatch * batch_size]
    
    # Reshape data into columns
    batchified_data = np.array(trimmed_data, dtype=np.int64).reshape((nbatch, batch_size), order='F')
    
    # Convert to Tensor if device and dtype are specified
    # if device or dtype:
    #     batchified_data = Tensor(batchified_data, device=device, dtype=dtype)
    
    return batchified_data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    # tot_seqlen = batches.shape[0]
    # assert i < tot_seqlen - 1
    # if i + bptt + 1 > tot_seqlen:
    #     X = batches[i : -1, :]
    #     y = batches[i+1 : , :].flatten()
    # else:
    #     X = batches[i : i + bptt, :]
    #     y = batches[i + 1: i + 1 + bptt, :].flatten()
    # return Tensor(X, device=device, dtype=dtype), Tensor(y, device=device, dtype=dtype)
    tot_seqlen = batches.shape[0]
    assert i < tot_seqlen - 1, "Index out of range for get_batch"

    # Slice data for the current batch
    data = batches[i : min(i + bptt, tot_seqlen - 1), :]

    # Slice target for the next time step and flatten
    target = batches[i + 1 : min(i + bptt + 1, tot_seqlen), :].flatten()

    # Convert to Tensor
    data_tensor = Tensor(data, device=device, dtype=dtype)
    target_tensor = Tensor(target, device=device, dtype=dtype)

    return data_tensor, target_tensor
    ### END YOUR SOLUTION