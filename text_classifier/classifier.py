import json
from text_classifier.functions import classify


# load our calculated synapse values
synapse_file = 'synapses.json'

with open(synapse_file) as data_file:
    synapse = json.load(data_file)

# classify("sudo make me a sandwich", synapses=synapses)
# classify("how are you today?", synapses=synapses)
# classify("talk to you tomorrow", synapses=synapses)
# classify("who are you?", synapses=synapses)
# classify("make me some lunch", synapses=synapses)
# classify("how was your lunch today?", synapses=synapses)
# print()
# classify("good day", synapses=synapses, show_details=True)

print()

classify("hi I am a robot from outer space", synapses=synapse, show_details=True)
