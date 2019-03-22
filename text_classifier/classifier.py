import json
import numpy as np
from text_classifier.base import think
from text_classifier.base import classes

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'

with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])
    synapses = (synapse_0, synapse_1)

def classify(sentence, synapses=None, show_details=False):

    print("Classifying sentence: {}".format(sentence))
    print()

    results = think(sentence, synapses, show_details)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("%s \n classification: %s" % (sentence, return_results))
    return return_results


# classify("sudo make me a sandwich", synapses=synapses)
# classify("how are you today?", synapses=synapses)
# classify("talk to you tomorrow", synapses=synapses)
# classify("who are you?", synapses=synapses)
# classify("make me some lunch", synapses=synapses)
# classify("how was your lunch today?", synapses=synapses)
# print()
# classify("good day", synapses=synapses, show_details=True)

print()

classify("hi I am a robot from outer space", synapses=synapses, show_details=True)
