## What is Word Embedding?
### Word Embeddings(Vectorization) are the texts converted into numbers and there may be different numerical representations of the same text.

<br />

### 1. One-Hot Encoding (OHE):
#### In this technique, we represent each unique word in vocabulary by setting a unique token with value 1 and rest 0 at other positions in the vector. In simple words, a vector representation of a one-hot encoded vector represents in the form of 1, and 0 where 1 stands for the position where the word exists and 0 everywhere else.

#### - Let’s consider the following sentence:


```
Sentence: I am teaching NLP in Python
```

#### A word in this sentence may be “NLP”, “Python”, “teaching”, etc.

#### Since a dictionary is defined as the list of all unique words present in the sentence. So, a dictionary may look like –

#### Therefore, the vector representation in this format according to the above dictionary is

```
Vector for NLP: [0,0,0,1,0,0] 
Vector for Python:  [0,0,0,0,0,1]
```

### - Disadvantages of One-hot Encoding:
1. #### the Size of the vector is equal to the count of unique words in the vocabulary.
2. #### One-hot encoding does not capture the relationships between different words. Therefore, it does not convey information about the context.

## Count Vectorizer:
1. ### It is one of the simplest ways of doing text vectorization.
2. ### It creates a document term matrix, which is a set of dummy variables that indicates if a particular word appears in the document.
3. ### Count vectorizer will fit and learn the word vocabulary and try to create a document term matrix in which the individual cells denote the frequency of that word in a particular document, which is also known as term frequency, and the columns are dedicated to each word in the corpus.