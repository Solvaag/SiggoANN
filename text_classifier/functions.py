import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import time

# probability threshold
ERROR_THRESHOLD = 0.2

stemmer = LancasterStemmer()


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def classify(sentence, synapses=None, show_details=False):
    classes = synapses['classes']

    print("Classifying sentence: {}".format(sentence))
    print()

    results = think(sentence, synapses, show_details)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("%s \n classification: %s" % (sentence, return_results))
    return return_results


def think(sentence, model, show_details=False):
    if model:
        synapse_0 = model["synapse0"]
        synapse_1 = model["synapse1"]
        words = model['words']
    else:
        raise ValueError("Missing model details.")

    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    input_layer = x
    # matrix multiplication of input and hidden layer
    inner_layer_0 = sigmoid(np.dot(input_layer, synapse_0))
    # output layer
    inner_layer_1 = sigmoid(np.dot(inner_layer_0, synapse_1))
    return inner_layer_1
