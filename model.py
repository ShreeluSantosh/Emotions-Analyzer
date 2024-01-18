#import tensorflow
#import tensorflow_datasets as tfds
import pandas as pd
import keras
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('goemotions_1.csv')
df = df.drop(columns = ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words(text):
  text.lower()
  text = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  text = [word for word in text if word.isalpha() and not word in stop_words]
  return ' '.join(text)

x = df.apply(lambda row: remove_stop_words(row['text']), axis=1)
y = df
y = y.drop(columns=['text', 'example_very_unclear'])

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
 
max_words = 20000
max_length = 500
 
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
x = pad_sequences(sequences, maxlen=max_length)

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential() 
model.add(Embedding(input_dim=20000, output_dim=100, input_length=max_length)) 
model.add(Flatten())
keras.layers.Dropout(rate=0.7)
model.add(Dense(128, activation='relu'))
keras.layers.Dropout(rate=0.65)
model.add(Dense(28, activation='relu'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.15, random_state=7)

model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=18)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")