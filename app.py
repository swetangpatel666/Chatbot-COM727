from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import random
from tensorflow.keras.models import load_model
import nltk
import pickle
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

intents_file = "intents2.json"
model_file = "chatbot_model.keras"

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
ignoreLetters = ['?', '!', '.', ',']

with open(intents_file) as file:
    intents = json.load(file)

model = load_model(model_file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if s:
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    message = request.form["message"]
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({"message": res})

if __name__ == "__main__":
    app.run(debug=True)
