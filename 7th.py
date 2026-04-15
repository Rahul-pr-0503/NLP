import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

def find_synonyms_and_antonyms(word):
    synsets = wn.synsets(word)
    synonyms = set()
    antonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    
    return list(synonyms), list(antonyms)
word = "active"
synonyms, antonyms = find_synonyms_and_antonyms(word)
print(f"Synonyms of '{word}': {synonyms}")
print(f"Antonyms of '{word}': {antonyms}")