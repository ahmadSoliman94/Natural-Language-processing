import spacy
from sklearn.feature_extraction.text import CountVectorizer

# # 1. generate n-grams using CountVectorizer:

# v = CountVectorizer() 
# v.fit(["Thor Hathodawala is looking for a job"])
# print(v.vocabulary_) # v.vocabulary_ is used to get the n-grams.


# # to get unigrams.
# v = CountVectorizer(ngram_range=(1,1)) 
# v.fit(["Thor Hathodawala is looking for a job"])
# print(v.vocabulary_)

# # to get bigrams.
# v = CountVectorizer(ngram_range=(1,2))
# v.fit(["Thor Hathodawala is looking for a job"])
# print(v.vocabulary_)

# # to get trigrams.
# v = CountVectorizer(ngram_range=(1,3))
# v.fit(["Thor Hathodawala is looking for a job"])
# print(v.vocabulary_)

##############################################################

# let's take a sample text documents, preprocess them to remove stop words, lemmatize them and then generate n-grams:



corpus = [
    "Thor ate pizza",
    "Loki is tall",
    "Loki is eating pizza"
] 

# load english language model.
nlp = spacy.load("en_core_web_sm") 


def preprocess(text):
    # remove stop words and lemmatize the text.
    doc = nlp(text)
    filterd_tokens = [] # to store the filtered tokens.
    
    # iterate over the tokens in the document.
    for token in doc:  
        if not token.is_stop or not token.is_punct: # check if the token is not a stop word or a punctuation.
            filterd_tokens.append(token.lemma_) # append the lemma of the token to the list.
    return " ".join(filterd_tokens)

print(preprocess("Thor ate pizza")) # output: Thor eat pizza.
print(preprocess("Loki is eating pizza")) # output: Loki eat pizza.


# preprocess the corpus.
corpus_processed = [
    preprocess(text) for text in corpus
]
print(corpus_processed) # output: ['Thor eat pizza', 'Loki tall', 'Loki eat pizza'].


v = CountVectorizer(ngram_range=(1,2)) # generate bigrams.
v.fit(corpus_processed)
print(v.vocabulary_) 

# Generate bag of n-gram vectors.
print(v.transform(["Thor eat pizza"]).toarray()) #  to get the bag of n-gram vectors.
