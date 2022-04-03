import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# run on first time to download punkt data
# nltk.download('punkt')

# initialize the stemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


def run():
    a = "How are you!?"
    words = ['hi', 'hey', 'how', 'are', 'you', 'is', 'anyon', 'there', 'hello', 'good', 'day', 'bye', 'see', 'you', 'later', 'goodby', 'thank', 'thank', 'you', 'that', "'s", 'help', 'thank', "'s", 'a', 'lot', 'which', 'item', 'do', 'you', 'have', 'what', 'kind', 'of', 'item', 'are', 'there', 'what', 'do', 'you', 'sell', 'do', 'you', 'take', 'credit', 'card', 'do', 'you', 'accept', 'mastercard', 'can', 'i', 'pay', 'with', 'paypal', 'are', 'you', 'cash', 'onli', 'how',
             'long', 'doe', 'deliveri', 'take', 'how', 'long', 'doe', 'ship', 'take', 'when', 'do', 'i', 'get', 'my', 'deliveri', 'tell', 'me', 'a', 'joke', 'tell', 'me', 'someth', 'funni', 'do', 'you', 'know', 'a', 'joke']
    bag = bag_of_words(tokenize(a), words)
    print(bag)
