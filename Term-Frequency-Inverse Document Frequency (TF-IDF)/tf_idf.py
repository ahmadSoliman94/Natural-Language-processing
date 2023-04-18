from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Create a corpus of documents
docs = [
  "The quick brown fox jumps over the lazy dog.",
  "A stitch in time saves nine.",
  "The early bird catches the worm.",
  "Actions speak louder than words.",
]

# Create a TfidfVectorizer object
tfidf = TfidfVectorizer()

# fit the vectorizer on the documents
tfidf.fit(docs)

# transform the documents into a document-term matrix
tfidf_matrix = tfidf.transform(docs)

# Convert the matrix to a pandas dataframe
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Display the dataframe
print(df_tfidf)
