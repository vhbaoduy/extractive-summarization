import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from nltk.tokenize.treebank import TreebankWordTokenizer



# Preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Tokenization function


def tokenize_text(text):
    tree_tokenizer = TreebankWordTokenizer()
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Tokenize each sentence into words
    words = [tree_tokenizer.tokenize(sentence, convert_parentheses=True) for sentence in sentences]
    return words


def detokenize_text(tokenized_text):
    # Join words in each sentence
    sentences = [' '.join(words) for words in tokenized_text]
    # Join sentences into a single text
    text = ' '.join(sentences)
    return text

# Reverse preprocessing function (simple, assuming lowercase conversion and HTML tag removal)


def restore_text(text):
    # Convert text to uppercase to restore original case (if needed)
    # text = text.upper()
    # Restore HTML tags (if needed)
    # For simplicity, assume HTML tags were removed and not stored
    return text
