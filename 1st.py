import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
def preprocess_text(text):
    tokens = word_tokenize(text)
    print("\nTokens:", tokens)
    tokens = [re.sub(r'[^A-Za-z]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]
    print("\nFiltered Tokens:", tokens)
    tokens = [token for token in tokens if token.isalpha()]
    print("\nScript Validated Tokens:", tokens)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    print("\nTokens after Stop Word Removal:", tokens)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    print("\nStemmed Tokens:", tokens)
    return tokens
text = "Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence! NLP helps computers understand human language."
processed_tokens = preprocess_text(text)
print("Final Processed Tokens:", processed_tokens)
