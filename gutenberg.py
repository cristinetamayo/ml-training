import requests
import spacy
import re
from nltk import FreqDist

def tokenize_text(text):
    nlp = spacy.load("es_core_news_sm")
    cleaned_text = re.sub(r"[^\w\s]", "", str(text))  # Remove non-alphanumeric characters
    cleaned_text = re.sub(r"rn+", "", cleaned_text)  # Remove occurrences of "rn" or "rnrn"
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    return tokens

def process_text_from_gutenberg(url):
    response = requests.get(url)
    text = response.text
    tokens = tokenize_text(text)
    return tokens

# Main program
if __name__ == "__main__":
    gutenberg_url = "https://www.gutenberg.org/cache/epub/53552/pg53552.txt"
    raw_text = process_text_from_gutenberg(gutenberg_url)
    preprocessed_text = tokenize_text(raw_text)

    # Now you have preprocessed_text as a list of words suitable for machine learning analysis
    # You can use this list for further processing or analysis like creating a bag-of-words or TF-IDF representation
    # Create frequency distribution
    freq_dist = FreqDist(preprocessed_text)

    # Print ten most common tokens
    print("Ten most common tokens:")
    for token, frequency in freq_dist.most_common(100):
        print(token, "-", frequency)