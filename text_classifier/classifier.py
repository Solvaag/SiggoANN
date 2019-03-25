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

review = "Hackers is a horror flick for everybody, that knows at least a little bit about computers. Young people, bursting with sex appeal are living almost underground and preaching rebel computer philosophy... They are hacking with gigabit connection, so that the nice animations of a broken fire walls and virtual rooms with databases and folders can load on their screens for the sake of the public. They use a laptops to hack from the top of the buildings, from the roofs, from the subways - because the laptop is such a powerful tool (in movies hackers almost always use laptops) - this device is meant to display \"ACCESS GRANTED\"! And the fight between the automated machines that change movies in the TV station office? Give me a freaking break! I rate this movie 2 for some shocking stupidity and frightening inadequacy."

classify(review, synapses=synapse)
