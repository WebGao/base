import unicodedata
import re
import random
import torch
import json

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 200


class Lang:
    """ Helper class to create and store dictionary of our vocabulary."""

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word.isdigit():
            for letter in word:
                if letter not in self.word2index:
                    self.word2index[letter] = self.n_words
                    self.word2count[letter] = 1
                    self.index2word[self.n_words] = letter
                    self.n_words += 1
                else:
                    self.word2count[letter] += 1
        else:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # Lowercase, trim, and remove non-letter characters.
    # s = unicode_to_ascii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"(([\u4e00-\u9fa5])+:|:([\u4e00-\u9fa5])+)", r"", s)
    return s


def read_langs(lang1, lang2, datapath, reverse=False):
    print("Reading lines...")
    lines = []
    # Read the file and split into lines.
    with open(datapath, encoding='utf-8') as f:

        for line in f:
            item = json.loads(line)
            if len(item['content'].split('\t')[1]) < 0:
                continue

            # answer = item['content'].split('\t')[2][2]
            #
            # begin = 2
            # if answer == '.':
            #     begin = 4
            lines.append(item['content'])
    # lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
    #     read().strip().split('\n')
    # Split every line into pairs and normalize.
    pairs = []
    for l in lines:
        item = l.split('\t')
        pairs.append([item[0], item[1]])
    # Reverse pairs, make Lang instances.
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(pair, reverse):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
           len(pair[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs, reverse):
    return [pair for pair in pairs if filter_pair(pair, reverse)]


def prepare_data(lang1, lang2, datapath, reverse):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, datapath, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs, reverse)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def indexes_from_sentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word.isdigit():
            for letter in word:
                indexes.append(lang.word2index[letter])
        else:
            indexes.append(lang.word2index[word])
    return indexes


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.insert(0, SOS_token)
    indexes.append(EOS_token)
    # return torch.tensor(indexes, dtype=torch.long).view(-1, 1)
    return indexes + [0] * (MAX_LENGTH - len(indexes))


def tensors_from_pair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def main():
    # Test data loading.
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', False)
    A = random.choice(pairs)
    print(tensors_from_pair(A, input_lang, output_lang))


if __name__ == "__main__":
    main()
