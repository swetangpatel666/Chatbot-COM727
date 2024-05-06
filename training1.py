import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import pickle

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents data
with open("intents2.json") as file:
    data = json.load(file)

# Initialize sets and lists for words, classes, and documents
ignore_letters = {"?", "!"}
words = set()

documents = []

# Process the intents to build words, classes, and documents
classes = set()

for intent in data['intents']:
    tag = intent['tag']
    if tag not in classes:
        classes.add(tag)

    for pattern in intent['patterns']:
        # Tokenize and lemmatize the words, then add them to the 'words' set
        tokenized_words = nltk.word_tokenize(pattern)
        lemmatized_words = {lemmatizer.lemmatize(w.lower()) for w in tokenized_words if w not in ignore_letters}
        words.update(lemmatized_words)

        # Add documents to the corpus
        documents.append((lemmatized_words, tag))

# Sort words and classes to maintain consistent ordering
words = sorted(set([lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]))
classes = sorted(set(classes))

# Save the words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []

# One-hot encode the classes
output_empty = [0] * len(classes)

for word_patterns, tag in documents:
    # Create a bag of words
    bag = [1 if word in word_patterns else 0 for word in words]

    # Create the output row for the class
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

# Shuffle and convert the training data to a NumPy array
random.shuffle(training)

# Explicitly extract the X and Y data
train_x = []
train_y = []

# Populate the training and output arrays
for item in training:
    if len(item) == 2:
        train_x.append(item[0])
        train_y.append(item[1])
    else:
        print("Inconsistent structure detected in training data")

# Convert lists to NumPy arrays with specific dtype
train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)

# Separate the training data into inputs and outputs
# train_x = training[:, 0]
# train_y = training[:, 1]

# Create the model architecture
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Configure the optimizer with appropriate learning rate, decay, and momentum
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model with a smaller batch size for stochastic learning
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("chatbot_model.keras")


# Model Accuracy:
loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
print("Accuracy of this model is:", accuracy)
