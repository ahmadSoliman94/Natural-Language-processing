import nltk

nltk.download()  # download the necessary dataset

text = "I'm doing fine, thanks./Not bad, thanks./Pretty good, thanks."

# tokenize the text into individual sentences
sentences = nltk.sent_tokenize(text)
tokens = nltk.word_tokenize(text)
print(sentences)
print(tokens)

