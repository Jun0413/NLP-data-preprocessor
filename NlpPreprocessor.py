###############################################################################
# author: jun0413
# date: 09/04/2018
#
###############################################################################

import os
import h5py
import json
import nltk
import itertools
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class NlpPreprocessor(object):

    def __init__(self, srcfile, tgtfile, data_type="classification", tokenize=False,
        TAGS={"start":"<S>", "end": "<E>", "pad": "<PAD>", "unk": "<UNK>"}):
        self.srcfile = srcfile
        self.tgtfile = tgtfile
        self.data_type = data_type
        if self.data_type == "classification":
            self.num_classes = len(open(tgtfile, "r").readlines())
            self.mappings = dict()
            self.gen_mappings()
        else:
            self.num_classes = -1
        self.tokenize = tokenize
        self.TAGS = TAGS

    def tokenise(self, sentence):
        return nltk.word_tokenize(sentence)

    """
    mapping: label_name -> label_idx
    save a copy of mapping
    """
    def gen_mappings(self):
        if self.data_type == "classification":
            label_set = set()
            for label in open(self.tgtfile, "r").readlines():
                label_set.add(label.strip())
            label_list = sorted(label_set) # in lexicographic order
            for i, label in enumerate(label_list):
                self.mappings[label] = i
            json.dump(self.mappings, open("mappings.json", "w"), indent=4)

    """
    for classification dataset
    return sentences and corresponding one-hot labels
    """
    def load_classification_data(self):
        sentences = [s.strip().lower() for s in open(self.srcfile, "r").readlines()]
        if self.tokenize == True:
            sentences = [self.tokenise(s) for s in sentences]
        else:
            sentences = [s.split(" ") for s in sentences]
        labels = []
        for label in open(self.tgtfile, "r").readlines():
            onehot_label = np.zeros(self.num_classes)
            onehot_label[self.mappings[label.strip()]] = 1
            labels.append(onehot_label)
        return [sentences, labels]

    """
    for seq2seq dataset
    return cleaned sentences respectively
    (each sentence is a *list*)
    """
    def load_seq2seq_data(self):
        src_sents = [s.strip().lower() for s in open(self.srcfile, "r").readlines()]
        tgt_sents = [s.strip().lower() for s in open(self.tgtfile, "r").readlines()]
        if self.tokenize == True:
            src_sents = [self.tokenise(s) for s in src_sents]
            tgt_sents = [self.tokenise(s) for s in tgt_sents]
        else:
            src_sents = [s.split(" ") for s in src_sents]
            tgt_sents = [s.split(" ") for s in tgt_sents]
        return [src_sents, tgt_sents]

    """
    sentences is a list of sentence lists
    - pad sentence to seq_len
    - chop off excess part
    - append sentence tags
    - mask sentences
    """
    def pad_sentences(self, sentences, seq_len, word2idx):
        pad_unk = True
        if word2idx is None:
            pad_unk = False
        padded_sentences = []
        masked_sentences = []
        for sentence in sentences:
            sent_len = len(sentence) + 2 # we want to append sentence tags after chopping
            if sent_len > seq_len:
                padded_sentence = [self.TAGS["start"]]+sentence[:(seq_len-2)]+[self.TAGS["end"]]
                masked_sentence = [1] * seq_len
            else:
                padded_sentence = [self.TAGS["start"]]+sentence+[self.TAGS["end"]]+(seq_len-sent_len)*[self.TAGS["pad"]]
                masked_sentence = [1] * sent_len + [0] * (seq_len-sent_len)
            # replace unknown words
            if pad_unk:
                vocabs = word2idx.keys()
                for i, word in enumerate(padded_sentence):
                    if word not in vocabs:
                        padded_sentence[i] = self.TAGS["unk"]

            padded_sentences.append(padded_sentence)
            masked_sentences.append(masked_sentence)

        return [masked_sentences, padded_sentences]

    """
    build vocabulary based on vocab_size or min_occur
    vocab inititates with all tags no matter what
    return idx2word and word2idx
    """
    def build_vocab(self, sentences, vocab_size=-1, min_occur=-1):
        vocab = [self.TAGS[key] for key in self.TAGS.keys()]
        word_counts = Counter(itertools.chain(*sentences))
        if vocab_size == -1 and min_occur == -1:
            vocab += [x[0] for x in word_counts.most_common()]
        elif min_occur == -1:
            vocab += [x[0] for x in word_counts.most_common(vocab_size)]
        else:
            vocab += [x[0] for x in word_counts.most_common(vocab_size) if x[1] >= min_occur]
        vocab = set(vocab)
        idx2word = {i: w for i, w in enumerate(vocab)}
        word2idx = {w: i for i, w in enumerate(vocab)}
        return [idx2word, word2idx]

    """
    for classification dataset
    return digitalized x and y
    """
    def classification_2idx(self, sentences, labels, word2idx):
        x = np.array([[word2idx[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]

    """
    for seq2seq dataset
    return digitalized x and y
    """
    def seq2seq_2idx(self, src_sents, tgt_sents, word2idx):
        x = np.array([[word2idx[word] for word in src] for src in src_sents])
        y = np.array([word2idx[word] for word in tgt] for tgt in tgt_sents)
        return [x, y]

    """
    use pretrained word2vec
    *assuming* the format is GoogleNews
    return idx2vec
    """
    def get_idx2vec(self, word2idx, word2vecfile):
        word2vec = {}
        rows = open(word2vecfile, "r").readlines()[1:]
        len_vector = len(rows[1].split()[1:]) # take arbitrary line to get vector length
        for row in rows: # omit header line
            values = row.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            word2vec[word] = vector
        idx2vec = np.zeros((len(word2idx.keys()), len_vector))
        for word, idx in word2idx.items():
            vector = word2vec.get(word)
            if vector is not None:
                idx2vec[idx] = vector
        del word2vec # release RAM
        return idx2vec

    """
    files to be saved:
    - idx2vec.h5
    - word2idx.json
    - idx2word.json
    will only save if files are not found
    """
    def save_vocab(self, word2idx, word2idx_name, idx2vec_name,
        idx2word, idx2word_name, word2vecfile):
        # save word2idx.json
        if not os.path.exists(word2idx_name): # DELETE THESE AFTER TESTING
            json.dump(word2idx, open(word2idx_name, "w"), indent=4)

        # save idx2word.json
        if not os.path.exists(idx2word_name):
            json.dump(idx2word, open(idx2word_name, "w"), indent=4)

        # save idx2vec.h5
        if not os.path.exists(idx2vec_name):
            idx2vec = self.get_idx2vec(word2idx, word2vecfile)
            h5f = h5py.File(idx2vec_name, "w")
            h5f.create_dataset("idx2vec", data=idx2vec)
            h5f.close()

    def load_word2idx(self, word2idx_name):
        return json.load(open(word2idx_name, "r"))

    def load_idx2word(self, idx2word_name):
        return json.load(open(idx2word_name, "r"))

    def load_idx2vec(self, idx2vec_name):
        h5f = h5py.File(idx2vec_name, "r")
        idx2vec = h5f["idx2vec"][:]
        h5f.close()
        return idx2vec

    """
    two subfunctions for build_input(*)
    """
    def build_classification_input(self, seq_len, word2idx, vocab_size, min_occur, load_vocab):
        sentences, labels = self.load_classification_data()
        sentences_masked, sentences_padded = self.pad_sentences(sentences, seq_len, word2idx=word2idx)
        if not load_vocab:
            idx2word, word2idx = self.build_vocab(sentences_padded, vocab_size, min_occur)
        else:
            idx2word = {v:k for k, v in word2idx.items()}
        x, y = self.classification_2idx(sentences_padded, labels, word2idx)
        return [x, y, idx2word, word2idx, sentences_masked]

    def build_seq2seq_input(self, seq_len, word2idx, vocab_size, min_occur, load_vocab):
        src_sents, tgt_sents = self.load_seq2seq_data()
        src_sents_masked, src_sents_padded = self.pad_sentences(src_sents, seq_len, word2idx=word2idx)
        tgt_sents_masked, tgt_sents_padded = self.pad_sentences(tgt_sents, seq_len, word2idx=word2idx)
        sentences_padded = src_sents_padded + tgt_sents_padded
        if not load_vocab:
            idx2word, word2idx = self.build_vocab(sentences_padded, vocab_size, min_occur)
        else:
            idx2word = {v:k for k, v in word2idx.items()}
        x, y = self.seq2seq_2idx(src_sents_padded, tgt_sents_padded, word2idx)
        return [x, y, idx2word, word2idx, src_sents_masked, tgt_sents_masked]

    """
    this is the **FINAL** function to call to load data
    - shuffle_seed: random seed for splitting train and test
    - test_size: (0~1) represents size of test set
    - vocab_size: maximum vocabulary, -1 means dontcare
    - min_occur: we only want words that appear (min_occur) times,
    and -1 means dontcare
    - seq_len: maximum length of a sentence, excess is cut off
    and dearth is padded
    - load_vocab: option for loading saved vocabulary, can be used
    when performing transfer learning
    - save_vocab: option for saving word2idx, idx2word and idx2vec
    - vocab_word2idx: word2idx.json to be loaded from or saved to
    - vocab_idx2word: idx2word.json to be loaded from or saved to
    - vocab_idx2vec: idx2vec.h5 to be loaded from or saved to
    - word2vecfile: pretrained word vectors, only support txt format
    like GoogleNews
    - mode: 'train' splits dataset, 'test' does not
    """
    def build_input(self, shuffle_seed=12, test_size=0.1, vocab_size=-1, min_occur=-1, seq_len=200,
        load_vocab=False, save_vocab=False, vocab_word2idx=None, vocab_idx2word=None, vocab_idx2vec=None,
        word2vecfile="GoogleNews", mode="train"):
        if vocab_size != -1 and min_occur != -1:
            raise Exception("Error: you can only confine vocab by either vocab_size or min_occur")
        if save_vocab:
            if os.path.exists(vocab_word2idx):
                raise Exception("Error: word2idx already exists, cannot be overwritten")
            if os.path.exists(vocab_idx2word):
                raise Exception("Error: idx2word already exists, cannot be overwritten")
            if os.path.exists(vocab_idx2vec):
                raise Exception("Error: idx2vec already exists, cannot be overwritten")
        if mode != "train" and mode != "test":
            raise Exception("Error: invalid mode, either train or test")

        word2idx = None
        if load_vocab:
            word2idx = self.load_word2idx(vocab_word2idx)
            idx2word = self.load_idx2word(vocab_idx2word)

        if self.data_type == "classification":
            x, y, idx2word, word2idx, sentences_masked\
                = self.build_classification_input(seq_len, word2idx, vocab_size, min_occur, load_vocab)
        elif self.data_type == "seq2seq":
            x, y, idx2word, word2idx, src_sents_masked, tgt_sents_masked\
                = self.build_seq2seq_input(seq_len, word2idx, vocab_size, min_occur, load_vocab)

        if save_vocab: # this is an expensive step
            print("saving vocabulary...")
            self.save_vocab(word2idx, vocab_word2idx, vocab_idx2vec, idx2word, vocab_idx2word, word2vecfile)

        if mode == "train":
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=shuffle_seed)
            if self.data_type == "classification":
                train_masked, val_masked = train_test_split(sentences_masked, test_size=test_size, random_state=shuffle_seed)
                return [X_train, X_val, y_train, y_val, idx2word, word2idx, train_masked, val_masked]
            elif self.data_type == "seq2seq":
                src_train_masked, src_val_masked, tgt_train_masked, tgt_val_masked\
                    = train_test_split(src_sents_masked, tgt_sents_masked, test_size=test_size, random_state=shuffle_seed)
                return [X_train, X_val, y_train, y_val, idx2word, word2idx, src_train_masked, src_val_masked,
                    tgt_train_masked, tgt_val_masked]
        elif mode == "test":
            if self.data_type == "classification":
                return [x, y, idx2word, word2idx, sentences_masked]
            elif self.data_type == "seq2seq":
                return [x, y, idx2word, word2idx, src_sents_masked, tgt_sents_masked]

    """
    return a generator zipping x and y
    """
    def batch_generator(self, x, y, batch_size, num_epochs):
        data = np.array(list(zip(x, y)))
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield data[start_index:end_index]

#=================================DEMO for how to use==================================================================

if __name__ == "__main__":
    # Phase 1: preprocess source dataset and save vocabs
    print("Phase 1\n"+60*"=")
    nlp_proc = NlpPreprocessor("demo_sents.txt", "demo_labels.txt", tokenize=True)
    X_train, X_test, y_train, y_test, idx2word, word2idx, train_masked, val_masked = nlp_proc.build_input(save_vocab=True, vocab_idx2vec="idx2vec.h5",
        vocab_word2idx="word2idx.json", vocab_idx2word="idx2word.json", word2vecfile="GoogleNews")
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    print(idx2word)
    print(word2idx)
    print(train_masked)
    print(val_masked)

    # Phase 2: use saved vocab to preprocess destination dataset
    print("Phase 2\n"+60*"=")
    nlp_proc = NlpPreprocessor("demo_2btransferred.txt", "demo_transferlabels.txt", tokenize=True)
    X_train, X_test, y_train, y_test, idx2word, word2idx, train_masked, val_masked = nlp_proc.build_input(load_vocab=True,
        vocab_word2idx="word2idx.json", vocab_idx2word="idx2word.json")
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    print(idx2word)
    print(word2idx)
    print(train_masked)
    print(val_masked)

    # Phase 3: demo on how to use batch generator
    print("Phase 3\n"+60*"=")
    train_batcher = nlp_proc.batch_generator(X_train, y_train, 2, 1)
    for batch in train_batcher:
        x_batch, y_batch = zip(*batch) # use an asterisk * to  unpack the containers
        print(x_batch)
        print(y_batch)
