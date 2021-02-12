#region #*Importing Dependencies
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
#endregion 

#region #*Loading Data
lem = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#endregion 

#region #*Creating Word Data Dictionary
words = [lem.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))                                                                          #?Set() Create set data and remove duplicaton

classes = sorted(set(classes))                                                                      #?No need of set() here

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
#endregion 

#region #*Bag of words
for document in documents:
    bag = []
    word_patterns = document[0]                                                                     #?[0]Word list [1]Class Name
    word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) 
    # print(bag)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1                                                      #?Making the answer class 1
    training.append([bag, output_row])                                                              #?adding hot encoded bag of word and answer to trainning
   
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
#endregion 

#region #*Constructing Model
model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
#endregion 

#region #*Save the Model
model.save('chatbot_brain.h5', hist)
print('Brain Cration Done!')
