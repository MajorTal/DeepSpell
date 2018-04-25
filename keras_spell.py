import os
import errno
from collections import Counter
import re
import json
import itertools
import logging
import requests
import numpy as np
import random

from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, recurrent
from keras.callbacks import Callback

# setup console logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

random.seed(123)  # Reproducibility


class Configuration(object):
    """Dump stuff here"""


CONFIG = Configuration()
CONFIG.input_layers = 2
CONFIG.output_layers = 2
CONFIG.amount_of_dropout = 0.2
CONFIG.hidden_size = 500
CONFIG.initialization = "he_normal"  # : Gaussian initialization scaled by fan-in (He et al., 2014)
CONFIG.number_of_chars = 100
CONFIG.max_input_len = 60
CONFIG.inverted = True

# parameters for the training:
CONFIG.batch_size = 100  # As the model changes in size, play with the batch size to best fit the process in memory
CONFIG.epochs = 500  # due to mini-epochs.
CONFIG.steps_per_epoch = 1000  # This is a mini-epoch. Using News 2013 an epoch would need to be ~60K.
CONFIG.validation_steps = 10
CONFIG.number_of_iterations = 10

# Parameters for the dataset
MIN_INPUT_LEN = 5
AMOUNT_OF_NOISE = 0.2 / CONFIG.max_input_len
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
PADDING = "☕"

DATA_FILES_PATH = "data"
DATA_FILES_FULL_PATH = os.path.join(os.path.dirname(__file__), DATA_FILES_PATH)
DATA_FILES_URL = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
NEWS_FILE_NAME_COMPRESSED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.shuffled.gz") # 1.1 GB
NEWS_FILE_NAME_ENGLISH = "news.2013.en.shuffled"
NEWS_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, NEWS_FILE_NAME_ENGLISH)
NEWS_FILE_NAME_CLEAN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.clean")
NEWS_FILE_NAME_FILTERED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.filtered")
NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split")
NEWS_FILE_NAME_TRAIN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.train")
NEWS_FILE_NAME_VALIDATE = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.validate")
CHAR_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "char_frequency.json")
SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_e{}.h5") # an HDF5 file

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769), chr(832),
                                                                                      chr(833), chr(2387), chr(5151),
                                                                                      chr(5152), chr(65344), chr(8242)),
                                  re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

# pylint:disable=invalid-name

def download_the_news_data():
    """Download the news data"""
    if not os.path.isfile(NEWS_FILE_NAME_COMPRESSED):
        logging.info("download_the_news_data: Downloading " + DATA_FILES_URL)
        try:
            os.makedirs(os.path.dirname(NEWS_FILE_NAME_COMPRESSED))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        with open(NEWS_FILE_NAME_COMPRESSED, "wb") as output_file:
            response = requests.get(DATA_FILES_URL, stream=True)
            total_length = response.headers.get('content-length')
            downloaded = percentage = 0
            print("»"*100)
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                downloaded += len(data)
                output_file.write(data)
                new_percentage = 100 * downloaded // total_length
                if new_percentage > percentage:
                    print("☑", end="")
                    percentage = new_percentage
        print()

def uncompress_data():
    """Uncompress the data files"""
    import gzip
    out_filename = NEWS_FILE_NAME_COMPRESSED[:-3]
    if not os.path.isfile(out_filename):
        logging.info("uncompress_data: " + out_filename)
        with gzip.open(NEWS_FILE_NAME_COMPRESSED, 'rb') as compressed_file:
            with open(out_filename, 'wb') as outfile:
                outfile.write(compressed_file.read())


def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    if random.random() < amount_of_noise * len(a_string):
        # Replace a character with a random character
        random_char_position = random.randint(0, len(a_string))
        a_string = a_string[:random_char_position] + random.choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if random.random() < amount_of_noise * len(a_string):
        # Delete a character
        random_char_position = random.randint(0, len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < CONFIG.max_input_len and random.random() < amount_of_noise * len(a_string):
        # Add a random character
        random_char_position = random.randint(0, len(a_string))
        a_string = a_string[:random_char_position] + random.choice(CHARS[:-1]) + a_string[random_char_position:]
    if random.random() < amount_of_noise * len(a_string):
        # Transpose 2 characters
        random_char_position = random.randint(0, len(a_string) - 2)
        a_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                    a_string[random_char_position + 2:])
    return a_string


def _vectorize(questions, answers, ctable):
    """Vectorize the data as numpy arrays"""
    len_of_questions = len(questions)
    X = np.zeros((len_of_questions, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    for i in range(len(questions)):
        sentence = questions.pop()
        for j, c in enumerate(sentence):
            try:
                X[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass # Padding
    y = np.zeros((len_of_questions, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    for i in range(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            try:
                y[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass # Padding
    return X, y


def slice_X(X, start=None, stop=None):
    """This takes an array-like, or a list of
    array-likes, and outputs:
        - X[start:stop] if X is an array-like
        - [x[start:stop] for x in X] if X in a list
    Can also work on list/array of indices: `slice_X(x, indices)`
    # Arguments
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    """
    if isinstance(X, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


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

    return X_train, X_val, y_train, y_val, CONFIG.max_input_len, ctable


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(CONFIG.input_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, input_shape=(None, len(chars)), kernel_initializer=CONFIG.initialization,
                                 return_sequences=layer_number + 1 < CONFIG.input_layers))
        model.add(Dropout(CONFIG.amount_of_dropout))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(CONFIG.output_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, return_sequences=True, kernel_initializer=CONFIG.initialization))
        model.add(Dropout(CONFIG.amount_of_dropout))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), kernel_initializer=CONFIG.initialization)))
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
        X = np.zeros((maxlen, len(self.chars)), dtype=np.bool) # pylint:disable=no-member
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X if x)


def generator(file_name):
    """Returns a tuple (inputs, targets)
    All arrays should contain the same number of samples.
    The generator is expected to loop over its data indefinitely.
    An epoch finishes when  samples_per_epoch samples have been seen by the model.
    """
    ctable = CharacterTable(read_top_chars())
    batch_of_answers = []
    while True:
        with open(file_name) as answers:
            for answer in answers:
                batch_of_answers.append(answer.strip())
                if len(batch_of_answers) == CONFIG.batch_size:
                    np.random.shuffle(batch_of_answers)
                    batch_of_questions = []
                    for answer_index, answer in enumerate(batch_of_answers):
                        question, answer = generate_question(answer)
                        batch_of_answers[answer_index] = answer
                        assert len(answer) == CONFIG.max_input_len
                        question = question[::-1] if CONFIG.inverted else question
                        batch_of_questions.append(question)
                    X, y = _vectorize(batch_of_questions, batch_of_answers, ctable)
                    yield X, y
                    batch_of_answers = []


def print_random_predictions(model, ctable, X_val, y_val):
    """Select 10 samples from the validation set at random so we can visualize errors"""
    print()
    try:
        for _ in range(10):
            ind = random.randint(0, len(X_val))
            rowX, rowy = X_val[np.asarray([ind])], y_val[np.asarray([ind])] # pylint:disable=no-member
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            if CONFIG.inverted:
                print('Q', q[::-1]) # inverted back!
            else:
                print('Q', q)
            print('A', correct)
            print(Colors.green + '☑' + Colors.close if correct == guess else Colors.red + '☒' + Colors.close, guess)
            print('---')
    except Exception:
        pass
    print()


class OnEpochEndCallback(Callback):
    """Execute this every end of epoch"""

    def on_epoch_end(self, epoch, logs=None):
        """On Epoch end - do some stats"""
        ctable = CharacterTable(read_top_chars())
        X_val, y_val = next(generator(NEWS_FILE_NAME_VALIDATE))
        print_random_predictions(self.model, ctable, X_val, y_val)
        self.model.save(SAVED_MODEL_FILE_NAME.format(epoch))


ON_EPOCH_END_CALLBACK = OnEpochEndCallback()


def iterative_train(model):
    """
    Iterative training of the model
     - To allow for finite RAM...
     - To allow infinite training data as the training noise is injected in runtime
    """
    model.fit_generator(generator(NEWS_FILE_NAME_TRAIN), steps_per_epoch=CONFIG.steps_per_epoch,
                        epochs=CONFIG.epochs,
                        verbose=1, callbacks=[ON_EPOCH_END_CALLBACK, ], validation_data=generator(NEWS_FILE_NAME_VALIDATE),
                        validation_steps=CONFIG.validation_steps,
                        class_weight=None, max_q_size=10, workers=1,
                        pickle_safe=False, initial_epoch=0)


def iterate_training(model, X_train, y_train, X_val, y_val, ctable):
    """Iterative Training"""
    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, CONFIG.number_of_iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=CONFIG.batch_size, epochs=CONFIG.epochs,
                  validation_data=(X_val, y_val))
        print_random_predictions(model, ctable, X_val, y_val)


def clean_text(text):
    """Clean the text - remove unwanted chars, fold punctuation etc."""
    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', text.strip())
    result = RE_DASH_FILTER.sub('-', result)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    result = RE_BASIC_CLEANER.sub('', result)
    return result


def preprocesses_data_clean():
    """Pre-process the data - step 1 - cleanup"""
    if not os.path.isfile(NEWS_FILE_NAME_CLEAN):
        logging.info("preprocesses_data_clean")
        with open(NEWS_FILE_NAME_CLEAN, "wt") as clean_data:
            for line in open(NEWS_FILE_NAME):
                clean_data.write(clean_text(line) + "\n")


def preprocesses_data_analyze_chars():
    """Pre-process the data - step 2 - analyze the characters"""
    if not os.path.isfile(CHAR_FREQUENCY_FILE_NAME):
        counter = Counter()
        logging.info("preprocesses_data_analyze_chars: Reading data:")
        for line in open(NEWS_FILE_NAME_CLEAN):
            counter.update(line)
    #     data = open(NEWS_FILE_NAME_CLEAN).read().decode('utf-8')
    #     logging.info("Read.\nCounting characters:")
    #     counter = Counter(data.replace("\n", ""))
        logging.info("Done.\nWriting to file:")
        with open(CHAR_FREQUENCY_FILE_NAME, 'wt') as output_file:
            output_file.write(json.dumps(counter))
        most_popular_chars = {key for key, _value in counter.most_common(CONFIG.number_of_chars)}
        logging.info("The top %s chars are:", CONFIG.number_of_chars)
        logging.info("".join(sorted(most_popular_chars)))


def read_top_chars():
    """Read the top chars we saved to file"""
    chars = json.loads(open(CHAR_FREQUENCY_FILE_NAME).read())
    counter = Counter(chars)
    most_popular_chars = {key for key, _value in counter.most_common(CONFIG.number_of_chars)}
    return most_popular_chars


def preprocesses_data_filter():
    """Pre-process the data - step 3 - filter only sentences with the right chars"""
    if not os.path.isfile(NEWS_FILE_NAME_FILTERED):
        most_popular_chars = read_top_chars()
        logging.info("preprocesses_data_filter: Reading and filtering data:")
        with open(NEWS_FILE_NAME_FILTERED, "wt") as output_file:
            for line in open(NEWS_FILE_NAME_CLEAN):
                decoded_line = line
                if decoded_line and not bool(set(decoded_line) - most_popular_chars):
                    output_file.write(line)
        logging.info("Done.")


def read_filtered_data():
    """Read the filtered data corpus"""
    logging.info("Reading filtered data:")
    lines = open(NEWS_FILE_NAME_FILTERED).read().split("\n")
    logging.info("Read filtered data - %s lines", len(lines))
    return lines


def preprocesses_split_lines():
    """Preprocess the text by splitting the lines between min-length and max_length
    I don't like this step:
      I think the start-of-sentence is important.
      I think the end-of-sentence is important.
      Sometimes the stripped down sub-sentence is missing crucial context.
      Important NGRAMs are cut (though given enough data, that might be moot).
    I do this to enable batch-learning by padding to a fixed length.
    """
    if not os.path.isfile(NEWS_FILE_NAME_SPLIT):
        logging.info("preprocesses_split_lines: Reading filtered data:")
        answers = set()
        with open(NEWS_FILE_NAME_SPLIT, "wt") as output_file:
            for _line in open(NEWS_FILE_NAME_FILTERED):
                line = _line
                while len(line) > MIN_INPUT_LEN:
                    if len(line) <= CONFIG.max_input_len:
                        answer = line
                        line = ""
                    else:
                        space_location = line.rfind(" ", MIN_INPUT_LEN, CONFIG.max_input_len - 1)
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
                    output_file.write(answer + "\n")


def preprocesses_split_lines2():
    """Preprocess the text by splitting the lines between min-length and max_length
    Alternative split.
    """
    if not os.path.isfile(NEWS_FILE_NAME_SPLIT):
        logging.info("Reading filtered data:")
        answers = set()
        for encoded_line in open(NEWS_FILE_NAME_FILTERED):
            line = encoded_line
            if CONFIG.max_input_len >= len(line) > MIN_INPUT_LEN:
                answers.add(line)
        logging.info("There are %s 'answers' (sub-sentences)", len(answers))
        logging.info("Here are some examples:")
        for answer in itertools.islice(answers, 10):
            logging.info(answer)
        with open(NEWS_FILE_NAME_SPLIT, "wt") as output_file:
            output_file.write("".join(answers))


def preprocesses_split_lines3():
    """Preprocess the text by selecting only max n-grams
    Alternative split.
    """
    if not os.path.isfile(NEWS_FILE_NAME_SPLIT):
        logging.info("Reading filtered data:")
        answers = set()
        for encoded_line in open(NEWS_FILE_NAME_FILTERED):
            line = encoded_line
            if line.count(" ") < 5:
                answers.add(line)
        logging.info("There are %s 'answers' (sub-sentences)", len(answers))
        logging.info("Here are some examples:")
        for answer in itertools.islice(answers, 10):
            logging.info(answer)
        with open(NEWS_FILE_NAME_SPLIT, "wt") as output_file:
            output_file.write("".join(answers))


# def preprocesses_split_lines4():
#     """Preprocess the text by selecting only sentences with most-common words AND not too long
#     Alternative split.
#     """
#     logging.info("Reading filtered data:")
#     from gensim.models.word2vec import Word2Vec
#     FILTERED_W2V = "fw2v.bin"
#     model = Word2Vec.load_word2vec_format(FILTERED_W2V, binary=True) # C text format
#     print(len(model.wv.index2word))
# #     answers = set()
# #     for encoded_line in open(NEWS_FILE_NAME_FILTERED):
# #         line = encoded_line.decode('utf-8')
# #         if line.count(" ") < 5:
# #             answers.add(line)
# #     logging.info("There are %s 'answers' (sub-sentences)", len(answers))
# #     logging.info("Here are some examples:")
# #     for answer in itertools.islice(answers, 10):
# #         logging.info(answer)
# #     with open(NEWS_FILE_NAME_SPLIT, "wt") as output_file:
# #         output_file.write("".join(answers))

def preprocess_partition_data():
    """Set asside data for validation"""
    if not os.path.isfile(NEWS_FILE_NAME_TRAIN):
        logging.info("preprocess_partition_data")
        answers = open(NEWS_FILE_NAME_SPLIT).read().split("\n")
        print('shuffle', end=" ")
        np.random.shuffle(answers)
        print("Done")
        # Explicitly set apart 10% for validation data that we never train over
        split_at = len(answers) - len(answers) // 10
        with open(NEWS_FILE_NAME_TRAIN, "wt") as output_file:
            output_file.write("\n".join(answers[:split_at]))
        with open(NEWS_FILE_NAME_VALIDATE, "wt") as output_file:
            output_file.write("\n".join(answers[split_at:]))


def generate_question(answer):
    """Generate a question by adding noise"""
    question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
    # Add padding:
    question += PADDING * (CONFIG.max_input_len - len(question))
    answer += PADDING * (CONFIG.max_input_len - len(answer))
    return question, answer


def generate_news_data():
    """Generate some news data"""
    print ("Generating Data")
    answers = open(NEWS_FILE_NAME_SPLIT).read().split("\n")
    questions = []
    print('shuffle', end=" ")
    np.random.shuffle(answers)
    print("Done")
    for answer_index, answer in enumerate(answers):
        question, answer = generate_question(answer)
        answers[answer_index] = answer
        assert len(answer) == CONFIG.max_input_len
        if random.randint(0, 100000) == 8:  # Show some progress
            print (len(answers))
            print ("answer:   '{}'".format(answer))
            print ("question: '{}'".format(question))
            print ()
        question = question[::-1] if CONFIG.inverted else question
        questions.append(question)

    return questions, answers


def train_speller_w_all_data():
    """Train the speller if all data fits into RAM"""
    questions, answers = generate_news_data()
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, y_train, y_val, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
    model = generate_model(y_maxlen, chars)
    iterate_training(model, X_train, y_train, X_val, y_val, ctable)


def train_speller(from_file=None):
    """Train the speller"""
    if from_file:
        logging.info("reloading existing model before training " + from_file)
        model = load_model(from_file)
    else:
        logging.info("training a new model")
        model = generate_model(CONFIG.max_input_len, chars=read_top_chars())
    iterative_train(model)


if __name__ == '__main__':
    download_the_news_data()  # download data/news.2013.en.shuffled.gz if dne
    uncompress_data()
    preprocesses_data_clean()
    preprocesses_data_analyze_chars()
    preprocesses_data_filter()
    preprocesses_split_lines()
#     preprocesses_split_lines2()
#     preprocesses_split_lines4()
    preprocess_partition_data()


    # get the latest epoch of any previous runs and use it as a starting model if it exists
    previous_models = [os.path.join(DATA_FILES_FULL_PATH, f) for f in os.listdir(DATA_FILES_FULL_PATH)
                       if os.path.isfile(os.path.join(DATA_FILES_FULL_PATH, f)) and f.startswith("keras_spell_e")]
    last_model = None
    latest_model_number = -1
    for model in previous_models:
        epoch = int(model.split("_")[-1].split(".")[0][1:])
        if epoch > latest_model_number:
            latest_model_number = epoch
            last_model = model

    # run with previous model or anew?
    if last_model is not None and os.path.isfile(last_model):
        train_speller(last_model)
    else:
        train_speller()
