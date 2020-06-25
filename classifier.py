#importing tensorflow
import tensorflow as tf
print(tf.__version__)

#importing tensorflow data services which holds a corpus of several datesets
import tensorflow_datasets as tfds 
#loading the imdb_reviews dataset which has 50,000 movie reviews, labeled either as positive or negative
#the dataset has been uploaded in the repository for reference
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

#the data is splitted into 25,000 reviews for training and 25,000 reviews for testing
import numpy as np

train_data, test_data = imdb['train'], imdb['test'] #splitting the training and testing/validation data

training_sentences = [] #list for training sentences
training_labels = [] #list for training labels (1:Positive review, 0:Negative review)

testing_sentences = [] #list for testing sentences
testing_labels = [] #list fro testing labels (1:Postive review, 0:Negative review)

#values for s and l are tensors so calling s.numpy() and l.numpy() will extract thier values
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8')) 
  training_labels.append(l.numpy())

#values for s and l are tensors so calling s.numpy() and l.numpy() will extract thier values
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print(training_sentences[0],end='\n')
print(training_labels_final[0])
#as visible the review is negative, hence the corresponding label too is negative i.e 0

#here we will tokenize and pad the sentences

vocab_size = 10000 #the maximum number of words to keep, based on word frequency
embedding_dim = 16 #dimension for the word embedding vectors
max_length = 120 #keeping the lengths of all sentences the same as 120
trunc_type='post' #truncating from the end of the sentences is sentences length is more than 120
oov_tok = "<OOV>" #designating the out of vocab token as "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) #creating the tokenizer object
tokenizer.fit_on_texts(training_sentences) #generating tokens by fitting on the training_sentences_list
word_index = tokenizer.word_index #generating a dictionary of words as key and the value as their tokens
sequences = tokenizer.texts_to_sequences(training_sentences) #generating the sequence of texts as their corresponding tokens
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type) #padding the sequences to make them all of the same length

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #reversing the word_index

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), #adding the embedding layer which has the shape of 120x16
    tf.keras.layers.Flatten(), #the values of embedding layers are flattened here (120*16=1920)
    tf.keras.layers.Dense(6, activation='relu'), #adding dense layer of 6 units
    tf.keras.layers.Dense(1, activation='sigmoid') #adding one output unit in the layer with a sigmoid activation as we are doing binary classification
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #compiling the model
model.summary()

num_epochs = 10 
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

#downloading the files for visualising word embeddings
import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n") 
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

#this block will download the vectors file which contains coefficients of the word vectors in the vecs.tsv file
#meta.tsv file will contain the word itself
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

#testing with own sentence
sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences([sentence])
padded = pad_sequences(sequence, maxlen=max_length, truncating=trunc_type)
#print(padded)
x = np.round(model.predict(padded))
print(x)
