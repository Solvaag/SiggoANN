import nltk
from nltk.stem.lancaster import LancasterStemmer
import os, json, datetime, csv
import numpy as np
import time
from text_classifier.functions import sigmoid, sigmoid_output_to_derivative

stemmer = LancasterStemmer()

packet = {'class': 'goof', 'sentence': 'how are you?'}

TRAINING_DATA = [{"class": "greeting", "sentence": "how are you?"},
                 {"class": "greeting", "sentence": "how is your day?"}, {"class": "greeting", "sentence": "good day"},
                 {"class": "greeting", "sentence": "how is it going today?"},
                 {"class": "goodbye", "sentence": "have a nice day"}, {"class": "goodbye", "sentence": "see you later"},
                 {"class": "goodbye", "sentence": "have a nice day"},
                 {"class": "goodbye", "sentence": "talk to you soon"},
                 {"class": "sandwich", "sentence": "make me a sandwich"},
                 {"class": "sandwich", "sentence": "can you make a sandwich?"},
                 {"class": "sandwich", "sentence": "having a sandwich today?"},
                 {"class": "sandwich", "sentence": "what's for lunch?"},
                 {'class': 'greeting', 'sentence': 'Hi, how are you?'}]


def build_bow_data(model=False):
    model_words = []
    model_classes = []
    documents = []
    ignore_words = ['?', ',', '\'', '\\']

    if model:
        model_words = model['words']
        model_classes = model['classes']

    # loop through each sentence in our training data
    for pattern in TRAINING_DATA:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern['sentence'])
        # add to our words list
        model_words.extend(w)
        # add to documents in our corpus
        documents.append((w, pattern['class']))
        # add to our classes list
        if pattern['class'] not in model_classes:
            model_classes.append(pattern['class'])

    # stem and lower each word and remove duplicates
    model_words = [stemmer.stem(w.lower()) for w in model_words if w not in ignore_words]
    model_words = list(set(model_words))

    # remove duplicates
    model_classes = list(set(model_classes))

    print(len(documents), "documents")
    print(len(model_classes), "classes", model_classes)
    print(len(model_words), "unique stemmed words", model_words)

    # create our training data
    result_training = []
    result_output = []
    # create an empty array for our output
    output_empty = [0] * len(model_classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in model_words:
            bag.append(1) if w in pattern_words else bag.append(0)

        result_training.append(bag)
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[model_classes.index(doc[1])] = 1
        result_output.append(output_row)

    # sample training/output
    i = 0
    w = documents[i][0]
    print([stemmer.stem(word.lower()) for word in w])
    print(result_training[i])
    print(result_output[i])

    if model:
        model['words'] = model_words
        model['classes'] = model_classes
    else:
        model = {}
        model['words'] = model_words
        model['classes'] = model_classes

    return result_training, result_output, model


def train(X, y, model=None, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    if model:
        classes = model['classes']
        words = model['words']

    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs + 1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                    1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if (j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("saved synapses to:", synapse_file)


def read_training_data(file_path):

    with open(file_path, encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        TRAINING_DATA.clear()
        count = 0
        top_count = 400
        for row in reader:
            if count >= top_count:
                break
            sentiment = int(row['sentiment'])

            if sentiment > 0:
                sentiment = "positive"
            else:
                sentiment = 'negative'

            packet = {'class':sentiment, 'sentence': row['review']}
            TRAINING_DATA.append(packet)
            count += 1
            print("Line {}".format(count))






if __name__ == '__main__':

    read_training_data("../data/labeledTrainData.tsv")

    try:
        with open('synapses.json', 'r') as model_file:
            model = json.load(model_file)
    except:
        model = None

    training, output, model = build_bow_data(model)

    X = np.array(training)
    y = np.array(output)

    start_time = time.time()

    train(X, y, model=model, hidden_neurons=40, alpha=0.1, epochs=400000, dropout=False, dropout_percent=0.2)

    elapsed_time = time.time() - start_time
    print("processing time:", elapsed_time, "seconds")
