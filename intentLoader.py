import json
  
json_data = None

def load_data(file_path):
    global json_data

    f = open(file_path, "r")
    content = f.read()
    data = json.loads(content)
    json_data = data
    return data

def get_classes():
    intents = json_data['intents']
    classes = []

    for i in range(len(intents)):
        intent = intents[i]
        classes.append(intent['tag'])

    return classes

def get_grouped():
    intents = json_data['intents']
    grouped_intents = []


    for i in range(len(intents)):
        intent = intents[i]
        class_name = intent['tag']
        #full_patterns = " ".join(intent['patterns'])
        patterns = intent['patterns']

        for j in range(len(patterns)):
            pattern = patterns[j]
            grouped = {'tag':class_name, 'pattern': pattern}

            grouped_intents.append(grouped)

    return grouped_intents
