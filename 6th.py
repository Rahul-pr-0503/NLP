import nltk
import nltk

nltk.download('brown')
from nltk.corpus import brown, inaugural, reuters, udhr, treebank
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag import UnigramTagger, RegexpTagger
print("Brown categories:", brown.categories())
print("Inaugural files:", inaugural.fileids())
print("Reuters categories:", reuters.categories()[:5])
print("UDHR languages sample:", udhr.fileids()[:5])
print("Sample raw, words, sentences from Brown:")
print(brown.raw()[:100])
print(brown.words(categories='news')[:20])
print(brown.sents(categories='news')[:2])
my_docs = {
    'greeting': ["hello world", "good morning"],
    'farewell': ["goodbye", "see you later"]
}
for cat, texts in my_docs.items():
    for t in texts:
        print(cat, nltk.word_tokenize(t))
pairs = [(genre, word.lower())
         for genre in ['news', 'romance']
         for word in brown.words(categories=genre)]
cfd = ConditionalFreqDist(pairs)
print("CFD conditions:", cfd.conditions())
cfd.tabulate(conditions=['news', 'romance'], samples=['the', 'love', 'can'])
tagged_sents = treebank.tagged_sents()[:100]
tagged_words = treebank.tagged_words()[:20]
print(tagged_sents[:2])
print(tagged_words[:10])
fd = FreqDist(tag for (_, tag) in tagged_words if tag.startswith('NN'))
print("Top noun tags:", fd.most_common(5))
word_props = {w: {'length': len(w), 'is_title': w.istitle()}
              for (w, _) in tagged_words[:20]}
print(word_props)
patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*s$', 'NNS'),
    (r'.*', 'NN')
]
regexp_tagger = RegexpTagger(patterns)
unigram = UnigramTagger(tagged_sents, backoff=regexp_tagger)
print(unigram.tag(["This", "is", "testing", "words", "running", "played"]))
vocab = set(w.lower() for w in brown.words())
def segment_text(s):
    results = {}
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            w = s[i:j]
            if w in vocab:
                results[w] = results.get(w, 0) + 1
    return results
squashed = "itisheday"
found = segment_text(squashed)
print("Segments found:", found)