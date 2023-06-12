from classifier import IntentClassifier

file_path = "./intents.json" #Folder where to store embeddings (instead of calculating them every time)
embeddings_folder = "./savedEmbeddings" #JSON file where intents are stored

my_classifier = IntentClassifier(file_path=file_path, save_folder=embeddings_folder, refresh=False)
# Use refresh=True when you have made changes to your intents.json file


#Example input
distance, class_name = my_classifier.classify("Turn on the lamp")
print(class_name, distance)


#Tester loops
query = input("> ")

while query!="exit":

    if query=="update":
        #Run this update function whenever you updated your intents.json file
        my_classifier.update()
        continue

    distance, class_name = my_classifier.classify(query)
    print(class_name, distance)
    query = input("> ")

print("Thanks for trying :))")