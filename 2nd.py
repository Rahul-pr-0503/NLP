import nltk
import re
from nltk import word_tokenize, ngrams
from collections import Counter
nltk.download('punkt')
corpus = """The Arabian Knights.
These are the fairy tales of the east. The stories of the Arabian knights are translated in many languages"""
def tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    words = text.lower().split()  
    return words
tokens = tokenize(corpus)
def compute_ngram_probabilities(tokens, n):
    """
    Computes n-gram probabilities using the conditional probability formula.
    For n = 1 (unigrams):
        P(word) = count(word) / total_word_count
    For n > 1:
        P(word_n | word_1, ..., word_(n-1)) = count(word_1, ..., word_n) / count(word_1, ..., word_(n-1))
    """
    ngrams_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngrams_list)
    if n == 1:
        total_count = sum(ngram_counts.values())
        probabilities = {ngram: count / total_count for ngram, count in ngram_counts.items()}
    else:
        prefix_counts = Counter(ngrams(tokens, n-1))
        probabilities = {}
        for ngram, count in ngram_counts.items():
            prefix = ngram[:-1]  
            probabilities[ngram] = count / prefix_counts[prefix]
    return probabilities
unigram_probs = compute_ngram_probabilities(tokens, 1)
bigram_probs  = compute_ngram_probabilities(tokens, 2)
trigram_probs = compute_ngram_probabilities(tokens, 3)
print("Unigram Probabilities:")
for ngram, prob in unigram_probs.items():
    print(f"P({ngram[0]}) = {prob:.4f}")
print("\nBigram Probabilities:")
for ngram, prob in bigram_probs.items():
    print(f"P({ngram[1]} | {ngram[0]}) = {prob:.4f}")
print("\nTrigram Probabilities:")
for ngram, prob in trigram_probs.items():
    print(f"P({ngram[2]} | {ngram[0]} {ngram[1]}) = {prob:.4f}")
