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
import itertools
import logging
import numpy as np
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, seed as random_seed, rand
from numpy import zeros as np_zeros # pylint:disable=no-name-in-module

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent
from keras.callbacks import Callback

# Set a logger for the module
LOGGER = logging.getLogger(__name__) # Every log will use the module name
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)

random_seed(123) # Reproducibility

# Parameters for the model and dataset
NUMBER_OF_ITERATIONS = 20000
EPOCHS_PER_ITERATION = 50
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
BATCH_SIZE = 18000 # As the model changes in size, play with the batch size to best fit the process in memory
SAMPLES_PER_EPOCH = 2000000
NUMBER_OF_VALIDATION_SAMPLES = 10000
HIDDEN_SIZE = 80
INITIALIZATION = "he_normal" # : Gaussian initialization scaled by fan-in (He et al., 2014)
MAX_INPUT_LEN = 20
MIN_INPUT_LEN = 5
INVERTED = True
AMOUNT_OF_NOISE = 0.2 / MAX_INPUT_LEN
NUMBER_OF_CHARS = 80
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")

DATA_FILES_PATH = "~/Downloads"
NEWS_FILE_NAME = "news.2011.en.shuffled"
NEWS_FILE_NAME_CLEAN = "news.2011.en.clean"
NEWS_FILE_NAME_FILTERED = "news.2011.en.filtered"
NEWS_FILE_NAME_SPLIT = "news.2011.en.split"
NEWS_FILE_NAME_TRAIN = "news.2011.en.train"
NEWS_FILE_NAME_VALIDATE = "news.2011.en.validate"
CHAR_FREQUENCY_FILE_NAME = "char_frequency.json"

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

def _vectorize(questions, answers, ctable):
    """Vectorize the data as numpy arrays"""
    len_of_questions = len(questions)
    X = np_zeros((len_of_questions, MAX_INPUT_LEN, ctable.size), dtype=np.bool)
    for i in xrange(len(questions)):
        sentence = questions.pop()
        for j, c in enumerate(sentence):
            X[i, j, ctable.char_indices[c]] = 1
    y = np_zeros((len_of_questions, MAX_INPUT_LEN, ctable.size), dtype=np.bool)
    for i in xrange(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            y[i, j, ctable.char_indices[c]] = 1
    return X, y

def vectorize(questions, answers, chars=None):
    """Vectorize the questions and expected answers"""
    print('Vectorization...')
    chars = chars or CHARS
    ctable = CharacterTable(chars)
    X, y = _vectorize(questions, answers, ctable)
    # Explicitly set apart 10% for validation data that we never train over
    split_at = int(len(X) - len(X) / 10)
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_val, y_train, y_val, MAX_INPUT_LEN, ctable


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
    green = '\033[92m'
    red = '\033[91m'
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

    @property
    def size(self):
        """The number of chars"""
        return len(self.chars)

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

def generator(file_name):
    """Returns a tuple (inputs, targets)
    All arrays should contain the same number of samples.
    The generator is expected to loop over its data indefinitely.
    An epoch finishes when  samples_per_epoch samples have been seen by the model.
    """
    ctable = CharacterTable(read_top_chars())
    batch_of_answers = []
    while True:
        with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, file_name))) as answers:
            for answer in answers:
                batch_of_answers.append(answer.strip().decode('utf-8'))
                if len(batch_of_answers) == BATCH_SIZE:
                    random_shuffle(batch_of_answers)
                    batch_of_questions = []
                    for answer_index, answer in enumerate(batch_of_answers):
                        question, answer = generate_question(answer)
                        batch_of_answers[answer_index] = answer
                        assert len(answer) == MAX_INPUT_LEN
                        question = question[::-1] if INVERTED else question
                        batch_of_questions.append(question)
                    X, y = _vectorize(batch_of_questions, batch_of_answers, ctable)
                    yield X, y
                    batch_of_answers = []

def print_random_predictions(model, ctable, X_val, y_val):
    """Select 10 samples from the validation set at random so we can visualize errors"""
    print()
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
        print(Colors.green + '☑' + Colors.close if correct == guess else Colors.red + '☒' + Colors.close, guess)
        print('---')
    print()


class OnEpochEndCallback(Callback):
    """Execute this every end of epoch"""

    def on_epoch_end(self, epoch, logs=None):
        """On Epoch end - do some stats"""
        ctable = CharacterTable(read_top_chars())
        X_val, y_val = next(generator(NEWS_FILE_NAME_VALIDATE))
        print_random_predictions(self.model, ctable, X_val, y_val)

ON_EPOCH_END_CALLBACK = OnEpochEndCallback()

def itarative_train(model):
    """Iterative training of the model"""
    model.fit_generator(generator(NEWS_FILE_NAME_TRAIN), samples_per_epoch=SAMPLES_PER_EPOCH, nb_epoch=EPOCHS_PER_ITERATION,
                        verbose=1, callbacks=[ON_EPOCH_END_CALLBACK, ], validation_data=generator(NEWS_FILE_NAME_VALIDATE),
                        nb_val_samples=NUMBER_OF_VALIDATION_SAMPLES,
                        class_weight=None, max_q_size=10, nb_worker=1,
                        pickle_safe=False, initial_epoch=0)


def iterate_training(model, X_train, y_train, X_val, y_val, ctable):
    """Iterative Training"""
    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, NUMBER_OF_ITERATIONS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS_PER_ITERATION, validation_data=(X_val, y_val))
        print_random_predictions(model, ctable, X_val, y_val)

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

def preprocesses_data_clean():
    """Pre-process the data - step 1 - cleanup"""
    LOGGER.info("Reading data:")
    news = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME))).read().decode('utf-8')
    LOGGER.info("Read the data\nCleaning data:")
    news = clean_text(news)
    LOGGER.info("Cleaned the data\nWriting to file:")
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_CLEAN)), "wb") as clean_news:
        clean_news.write(news.encode("utf-8"))
    LOGGER.info("Written to file")

def preprocesses_data_analyze_chars():
    """Pre-process the data - step 2 - analyze the characters"""
    LOGGER.info("Reading data:")
    data = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_CLEAN))).read().decode('utf-8')
    LOGGER.info("Read.\nCounting characters:")
    counter = Counter(data.replace("\n", ""))
    LOGGER.info("Done.\nWriting to file:")
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, CHAR_FREQUENCY_FILE_NAME)), 'wb') as output_file:
        output_file.write(json.dumps(counter))
    most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
    LOGGER.info("The top %s chars are:", NUMBER_OF_CHARS)
    LOGGER.info("".join(sorted(most_popular_chars)))

def read_top_chars():
    """Read the top chars we saved to file"""
    chars = json.loads(open(os.path.expanduser(os.path.join(DATA_FILES_PATH, CHAR_FREQUENCY_FILE_NAME))).read())
    counter = Counter(chars)
    most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
    return most_popular_chars

def preprocesses_data_filter():
    """Pre-process the data - step 3 - filter only sentences with the right chars"""
    most_popular_chars = read_top_chars()
    LOGGER.info("Reading data:")
    data = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_CLEAN))).read().decode('utf-8')
    LOGGER.info("Read.\nFiltering:")
    lines = [line.strip() for line in data.split('\n')]
    LOGGER.info("Read %s lines of input corpus", len(lines))
    lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
    LOGGER.info("Left with %s lines of input corpus", len(lines))
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_FILTERED)), "wb") as output_file:
        output_file.write("\n".join(lines).encode('utf-8'))

def read_filtered_data():
    """Read the filtered data corpus"""
    LOGGER.info("Reading filtered data:")
    lines = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_FILTERED))).read().decode('utf-8').split("\n")
    LOGGER.info("Read filtered data - %s lines", len(lines))
    return lines

def preprocesses_split_lines():
    """Preprocess the text by splitting the lines between min-length and max_length"""
    LOGGER.info("Reading filtered data:")
    lines = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_FILTERED))).read().decode('utf-8').split("\n")
    LOGGER.info("Read filtered data - %s lines", len(lines))
    answers = set()
    while lines:
        line = lines.pop()
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
            answers.add(answer)
    LOGGER.info("there are %s 'answers' (sub-sentences)", len(answers))
    for answer in itertools.islice(answers, 10):
        LOGGER.info(answer)
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_SPLIT)), "wb") as output_file:
        output_file.write("\n".join(answers).encode('utf-8'))

def preprocess_partition_data():
    """Set asside data for validation"""
    answers = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_SPLIT))).read().decode('utf-8').split("\n")
    print('shuffle', end=" ")
    random_shuffle(answers)
    print("Done")
    # Explicitly set apart 10% for validation data that we never train over
    split_at = len(answers) - len(answers) // 10
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_TRAIN)), "wb") as output_file:
        output_file.write("\n".join(answers[:split_at]).encode('utf-8'))
    with open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_VALIDATE)), "wb") as output_file:
        output_file.write("\n".join(answers[split_at:]).encode('utf-8'))


def generate_question(answer):
    """Generate a question by adding noise"""
    question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
    # Add padding:
    question += '.' * (MAX_INPUT_LEN - len(question))
    answer += "." * (MAX_INPUT_LEN - len(answer))
    return question, answer

def generate_news_data():
    """Generate some news data"""
    print ("Generating Data")
    answers = open(os.path.expanduser(os.path.join(DATA_FILES_PATH, NEWS_FILE_NAME_SPLIT))).read().decode('utf-8').split("\n")
    questions = []
    print('shuffle', end=" ")
    random_shuffle(answers)
    print("Done")
    for answer_index, answer in enumerate(answers):
        question, answer = generate_question(answer)
        answers[answer_index] = answer
        assert len(answer) == MAX_INPUT_LEN
        if random_randint(100000) == 8: # Show some progress
            print (len(answers))
            print ("answer:   '{}'".format(answer))
            print ("question: '{}'".format(question))
            print ()
        question = question[::-1] if INVERTED else question
        questions.append(question)

    return questions, answers

def main_news():
    """Main"""
    questions, answers = generate_news_data()
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, y_train, y_val, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
    model = generate_model(y_maxlen, chars)
    iterate_training(model, X_train, y_train, X_val, y_val, ctable)

def train_speller():
    """Train the speller"""
    model = generate_model(MAX_INPUT_LEN, chars=read_top_chars())
    itarative_train(model)
#     iterate_training(model, X_train, y_train, X_val, y_val, ctable)

if __name__ == '__main__':
#     preprocesses_data_clean()
#     preprocesses_data_analyze_chars()
#     preprocesses_data_filter()
#     preprocesses_split_lines()
#     preprocess_partition_data()
    train_speller()
