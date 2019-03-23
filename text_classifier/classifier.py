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

review = "National Socialism (German: Nationalsozialismus), more commonly known as Nazism (/ˈnɑːtsiɪzəm, ˈnæt-/),[1] is the ideology and practices associated with the Nazi Party – officially the National Socialist German Workers' Party (Nationalsozialistische Deutsche Arbeiterpartei or NSDAP) – in Nazi Germany, and of other far-right groups with similar aims. Nazism is a form of fascism and showed that ideology's disdain for liberal democracy and the parliamentary system, but also incorporated fervent antisemitism, scientific racism, and eugenics into its creed. Its extreme nationalism came from Pan-Germanism and the Völkisch movement prominent in the German nationalism of the time, and it was strongly influenced by the anti-Communist Freikorps paramilitary groups that emerged after Germany's defeat in World War I, from which came the party's 'cult of violence' which was 'at the heart of the movement.'[2] Nazism subscribed to theories of racial hierarchy and Social Darwinism, identifying the Germans as a part of what the Nazis regarded as an Aryan or Nordic master race.[3] It aimed to overcome social divisions and create a German homogeneous society based on racial purity which represented a people's community (Volksgemeinschaft). The Nazis aimed to unite all Germans living in historically German territory, as well as gain additional lands for German expansion under the doctrine of Lebensraum and exclude those who they deemed either community aliens or 'inferior' races."

classify(review, synapses=synapse, show_details=True)
