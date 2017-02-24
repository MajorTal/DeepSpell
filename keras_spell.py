# encoding: utf-8
'''
Created on Nov 26, 2015

@author: tal

Based in part on:
Learn math - https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py

See https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm
'''

from __future__ import print_function, division, unicode_literals

import os
from collections import Counter
import re
import json
import numpy as np
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, seed as random_seed, rand
from numpy import zeros as np_zeros # pylint:disable=no-name-in-module

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent

random_seed(123) # Reproducibility

# Parameters for the model and dataset
NUMBER_OF_ITERATIONS = 20000
EPOCHS_PER_ITERATION = 5
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
BATCH_SIZE = 500
HIDDEN_SIZE = 700
INITIALIZATION = "he_normal" # : Gaussian initialization scaled by fan_in (He et al., 2014)
MAX_INPUT_LEN = 40
MIN_INPUT_LEN = 3
INVERTED = True
AMOUNT_OF_NOISE = 0.2 / MAX_INPUT_LEN
NUMBER_OF_CHARS = 100 # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")

DATA_FILES_PATH = "~/Downloads"
NEWS_FILE_NAME = "news.2011.en.shuffled"
CLEAN_NEWS_FILE_NAME = "clean_news.2011.en.shuffled"
FILTERED_NEWS_FILE_NAME = "filtered_news.2011.en.shuffled"
MOST_POPULAR_CHARS_FILE_NAME = "most_popular_chars.json"

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(unichr(768), unichr(769), unichr(832),
                                                                                      unichr(833), unichr(2387), unichr(5151),
                                                                                      unichr(5152), unichr(65344), unichr(8242)),
                                  re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

# pylint:disable=invalid-name

def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    if rand() < amount_of_noise * len(a_string):
        # Replace a character with a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if rand() < amount_of_noise * len(a_string):
        # Delete a character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < MAX_INPUT_LEN and rand() < amount_of_noise * len(a_string):
        # Add a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
    if rand() < amount_of_noise * len(a_string):
        # Transpose 2 characters
        random_char_position = random_randint(len(a_string) - 1)
        a_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                    a_string[random_char_position + 2:])
    return a_string



def vectorize(questions, answers, chars=None):
    """Vectorize the questions and expected answers"""
    print('Vectorization...')
    chars = chars or CHARS
    x_maxlen = max(len(question) for question in questions)
    y_maxlen = max(len(answer) for answer in answers)
#     print (len(questions), x_maxlen, len(chars))
    len_of_questions = len(questions)
    ctable = CharacterTable(chars)
    print("X = np_zeros")
    X = np_zeros((len_of_questions, x_maxlen, len(chars)), dtype=np.bool)
    print("for i, sentence in enumerate(questions):")
    for i in xrange(len(questions)):
        sentence = questions.pop()
        for j, c in enumerate(sentence):
            X[i, j, ctable.char_indices[c]] = 1
    print("y = np_zeros")
    y = np_zeros((len_of_questions, y_maxlen, len(chars)), dtype=np.bool)
    print("for i, sentence in enumerate(answers):")
    for i in xrange(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            y[i, j, ctable.char_indices[c]] = 1

    # Explicitly set apart 10% for validation data that we never train over
    split_at = int(len(X) - len(X) / 10)
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_val, y_train, y_val, y_maxlen, ctable


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(INPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(None, len(chars)), init=INITIALIZATION,
                                 return_sequences=layer_number + 1 < INPUT_LAYERS))
        model.add(Dropout(AMOUNT_OF_DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(OUTPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, return_sequences=True, init=INITIALIZATION))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), init=INITIALIZATION)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class Colors(object):
    """For nicer printouts"""
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool) # pylint:disable=no-member
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def iterate_training(model, X_train, y_train, X_val, y_val, ctable):
    """Iterative Training"""
    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, NUMBER_OF_ITERATIONS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS_PER_ITERATION, validation_data=(X_val, y_val))
        # Select 10 samples from the validation set at random so we can visualize errors
        for _ in range(10):
            ind = random_randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])] # pylint:disable=no-member
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            if INVERTED:
                print('Q', q[::-1]) # inverted back!
            else:
                print('Q', q)
            print('A', correct)
            print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
            print('---')

def clean_text(text):
    """Clean the text - remove unwanted chars, fold punctuation etc."""
    from time import time
    start_time = time()
    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', text.strip())
    print("NORMALIZE_WHITESPACE_REGEX", time() - start_time)
    result = RE_DASH_FILTER.sub('-', result)
    print("RE_DASH_FILTER", time() - start_time)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    print("RE_APOSTROPHE_FILTER", time() - start_time)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    print("RE_LEFT_PARENTH_FILTER")
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    print("RE_RIGHT_PARENTH_FILTER")
    result = RE_BASIC_CLEANER.sub('', result)
    print("RE_BASIC_CLEANER")
    return result

def preprocesses_data1():
    """Pre-process the data - step 1"""
    print("Reading data:")
    news = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME))).read().decode('utf-8')
    print("Read the data\nCleaning data:")
    news = clean_text(news)
    print("Cleaned the data\nWriting to file:")
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, CLEAN_NEWS_FILE_NAME)), "wb") as clean_news:
        clean_news.write(news.encode("utf-8"))
    print("Written to file")

def preprocesses_data2():
    """Pre-process the data - step 2"""
    print("Reading data:")
    data = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, CLEAN_NEWS_FILE_NAME))).read().decode('utf-8')
    print("Read.\nCounting characters:")
    counter = Counter(data.replace("\n", ""))
    print("Done.\nWriting to file:")
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, MOST_POPULAR_CHARS_FILE_NAME)), 'wb') as output_file:
        output_file.write(json.dumps(counter))
    print("Done")

def preprocesses_data3():
    """Pre-process the data - step 3"""
    chars = json.loads(open(os.path.expanduser(os.path.join(DATA_FILES_PATH, MOST_POPULAR_CHARS_FILE_NAME))).read())
    counter = Counter(chars)
    most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
    print("The top {} chars are:".format(NUMBER_OF_CHARS))
    print("".join(sorted(most_popular_chars)))
    print("Reading data:")
    data = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, CLEAN_NEWS_FILE_NAME))).read().decode('utf-8')
    print("Read.\nFiltering:")
    lines = [line.strip() for line in data.split('\n')]
    print("Read {} lines of input corpus".format(len(lines)))
    lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
    print("Left with {} lines of input corpus".format(len(lines)))
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, FILTERED_NEWS_FILE_NAME)), "wb") as output_file:
        output_file.write("\n".join(lines).encode('utf-8'))

def read_news():
    """Read the news corpus"""
    print("reading news")
    lines = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, FILTERED_NEWS_FILE_NAME))).read().decode('utf-8').split("\n")
    print("read news")
    return lines



def generate_news_data(corpus):
    """Generate some news data"""
    print ("Generating Data")
    questions, answers, seen_answers = [], [], set()
    while corpus:
        line = corpus.pop()
        while len(line) > MIN_INPUT_LEN:
            if len(line) <= MAX_INPUT_LEN:
                answer = line
                line = ""
            else:
                space_location = line.rfind(" ", MIN_INPUT_LEN, MAX_INPUT_LEN - 1)
                if space_location > -1:
                    answer = line[:space_location]
                    line = line[len(answer) + 1:]
                else:
                    space_location = line.rfind(" ") # no limits this time
                    if space_location == -1:
                        break # we are done with this line
                    else:
                        line = line[space_location + 1:]
                        continue
            if answer and answer in seen_answers:
                continue
            seen_answers.add(answer)
            answers.append(answer)
        if random_randint(100000) == 8: # Show some progress
            print('.', end="")
    print('shuffle', end=" ")
    random_shuffle(answers)
    print("Done")
    for answer_index, answer in enumerate(answers):
        question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
        question += '.' * (MAX_INPUT_LEN - len(question))
        answer += "." * (MAX_INPUT_LEN - len(answer))
        answers[answer_index] = answer
        assert len(answer) == MAX_INPUT_LEN
        if random_randint(100000) == 8: # Show some progress
            print (len(seen_answers))
            print ("answer:   '{}'".format(answer))
            print ("question: '{}'".format(question))
            print ()
        question = question[::-1] if INVERTED else question
        questions.append(question)

    return questions, answers

def main_news():
    """Main"""
    questions, answers = generate_news_data(read_news())
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, y_train, y_val, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
    model = generate_model(y_maxlen, chars)
    iterate_training(model, X_train, y_train, X_val, y_val, ctable)

if __name__ == '__main__':
    main_news()
