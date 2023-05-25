import re 
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report




# ==================== Yelp Reviews Classifier ==================== # 

class Classifier:
    def __init__(self, df) -> None:
        self.df = df

    def read_data(self):
        """
        Read data from csv file
        """
        return pd.read_csv(self.df)
    
    def show_data(self, df):
        """
        Show the data
        """
        print(df.head()) 

    def get_yelp_class(self, df):
        """
        Get the yelp class
        """
        yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]
        return yelp_class 
    
    def get_X_y(self, yelp_class):
        """
        Get the X and y
        """
        X = yelp_class['text'].values
        y = yelp_class['stars'].values
        return X, y
    
    def clean_text(self, text):
        """
        Clean the text
        """
        # Remove all the punctuation for example: !, @, #, $, %, ^, &, *, (, ), -, _, +, =, {, }, [, ], |, \, :, ;, ", ', <, >, ?, /, .
        corpus = [] 
        for i in range(0, len(text)):
            review = re.sub('[^a-zA-Z]', ' ', text[i])
            review = review.lower() # Convert all the letters to lower case
            review = nltk.word_tokenize(review) # Convert the sentence to a list of words
            ps = PorterStemmer() # Stemming
            all_stopwords = stopwords.words('english') # Remove all the stopwords
            all_stopwords.remove('not') # Remove the word 'not' from the stopwords
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # Stemming and remove all the stopwords
            review = ' '.join(review) # Join the words to form a sentence
            corpus.append(review) # Append the sentence to the corpus
        return corpus
    
    def vectorize(self, corpus):
        """
        Vectorize the corpus
        """
        cv = CountVectorizer()
        X = cv.fit_transform(corpus).toarray()
        print(f'The number of the unique words: {len(cv.get_feature_names_out())}') # Get the number of unique words
        df = pd.DataFrame(X, columns=cv.get_feature_names_out()) # Show the vectorized corpus in a dataframe
        return X, df   # Return the vectorized corpus and the dataframe
    
    def train_test_split(self, X, y):
        """
        Split the data into training set and testing set
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        print(f'The shape of the training set: {X_train.shape}', f'The shape of the testing set: {X_test.shape}')
        print(f'The shape of the training set: {y_train.shape}', f'The shape of the testing set: {y_test.shape}')
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the model using Random Forest Classifier
        """
        rf = RandomForestClassifier(n_estimators=200)
        rf.fit(X_train, y_train)
        return rf
    
    def predict(self, rf, X_test):
        """
        Predict the testing set
        """
        y_pred = rf.predict(X_test)
        return y_pred
    
    def evaluate(self, y_test, y_pred):
        """
        Evaluate the model
        """
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        
    



    





# ==================== Main Function ==================== #
if __name__ == '__main__':

    # Initialize the classifier
    clf = Classifier('./yelp.csv')

    # Get the data
    df = clf.read_data()

    # Show the data
    #clf.show_data(df)

    # Get the yelp class
    yelp_class = clf.get_yelp_class(df)
    print(yelp_class.head())

    print('====================================================')

    # Get the X and y
    X, y = clf.get_X_y(yelp_class)
    #   print(X)

    # Clean the text
    corpus = clf.clean_text(X)

    # Vectorize the corpus
    X, df = clf.vectorize(corpus)
    print(df.head())
    print("====================================================")

    # Split the data into training set and testing set
    X_train, X_test, y_train, y_test = clf.train_test_split(X, y)
    print("====================================================")
    
    # Train the model
    rf = clf.train_model(X_train, y_train)

    # Predict the testing set
    y_pred = clf.predict(rf, X_test)

    # Evaluate the model
    clf.evaluate(y_test, y_pred)

    

    