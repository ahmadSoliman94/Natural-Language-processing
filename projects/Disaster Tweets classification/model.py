import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import gensim
from gensim.models import Word2Vec
 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,  KFold,StratifiedKFold, GridSearchCV, RandomizedSearchCV,cross_val_score
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  RobustScaler

import xgboost as xgb





# ==================== Disasters Tweets Classification ==================== #


"""
About the dataset:
- predicting whether a given tweet is about a real disaster or not. 

Columns:
- id - a unique identifier for each tweet.
- text - the text of the tweet.
- location - the location the tweet was sent from (may be blank).
- keyword - a particular keyword from the tweet (may be blank).
- target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0).
"""    

class DisasterTweetsClassification:
    def __init__(self, dataframe_path) -> None:
        self.dataframe_path = dataframe_path
    
    def read_data(self, file_path):
        """
        Read the data from the csv file and return a dataframe.
        """
        df = pd.read_csv(file_path)
        return df
    
    def show_shape(self, dataframe):
        """
        Show the shape of the dataframe.
        """
        # to print the name of the dataframe and its shape
        print(f" the Shape of {dataframe.name} : {dataframe.shape}")

    def show_info(self, dataframe):
        """
        Show the info of the dataframe.
        """
        # to print the name of the dataframe and its info
        print(dataframe.info())

    
    def show_missing_values(self, dataframe):
        """
        Show the missing values of the dataframe.
        """
        # to print the name of the dataframe and its missing values
        print(f"Missing values in {dataframe.name}:\n")
        print(dataframe.isnull().sum())
        print("\n")

    def plot_top_keywords(self, dataframe):
        """
        Plot a barplot of the top 20 keywords in the dataframe.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(y=dataframe['keyword'].value_counts()[:20].index,
                    x=dataframe['keyword'].value_counts()[:20],
                    orient='h')
        plt.xlabel('Count')
        plt.ylabel('Keyword')
        plt.title('Top 20 Keywords')
        plt.savefig('./figure/top_keywords_plot.png')  # Save the plot as an image file

    def plot_top_locations(self, dataframe):
        """
        Plot a barplot of the top 5 locations in the dataframe.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(y=dataframe['location'].value_counts()[:5].index,
                    x=dataframe['location'].value_counts()[:5],
                    orient='h')
        plt.xlabel('Count')
        plt.ylabel('Location')
        plt.title('Top 5 Locations')
        plt.savefig('./figure/top_locations_plot.png')

    def drop_columns(self, dataframe, columns):
        """
        Drop the columns from the dataframe.
        """
        dataframe.drop(columns=columns, inplace=True)
        return dataframe

    def plot_disaster_count(self,dataframe):
        """
        Plot a countplot for the target variable in the dataframe with labeled bars.
        """
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=dataframe, x='target', palette='RdPu_r')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title('Countplot for Disaster and Non-disaster Related Tweets')
        plt.savefig('./figure/disaster_count_plot.png')


    def get_X_y(self, yelp_class):
        """
        Get the X and y
        """
        X = yelp_class['text'].values
        y = yelp_class['target'].values
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
        cv = TfidfVectorizer()
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
    
    # train multiple models and compare them 
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models and compare them
        """
        # Create a dictionary to store the models
        models = {
                  'Random Forest': RandomForestClassifier(),
                  'xgboost': xgb.XGBClassifier()
                  }
        
        # Create a function to fit and score the models
        def fit_and_score(models, X_train, X_test, y_train, y_test):
            """
            Fit and score the models
            """
            # Set a random seed
            np.random.seed(42)
            # Make a dictionary to keep model scores
            model_scores = {}
            # Loop through the models
            for name, model in models.items():
                # Fit the model to the data
                model.fit(X_train, y_train)
                # Evaluate the model and append its score to model_scores
                model_scores[name] = model.score(X_test, y_test)

                # plot confusion matrix using seaborn's heatmap() and save the plot as an image file
                fig, ax = plt.subplots(figsize=(10, 6))
                ax = sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, cbar=False, fmt=',d')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix for {name}')
                plt.savefig(f'./figure/{name}_confusion_matrix.png')

                # print classification report and save the report as a text file
                print(f'Classification Report for {name}')
                print(classification_report(y_test, model.predict(X_test)))
                with open(f'./figure/{name}_classification_report.txt', 'w') as f:
                    f.write(f'Classification Report for {name}\n')
                    f.write(classification_report(y_test, model.predict(X_test)))

            return model_scores
        
        
        # Get the scores of the models
        model_scores = fit_and_score(models=models,
                                     X_train=X_train,
                                     X_test=X_test,
                                     y_train=y_train,
                                     y_test=y_test)
        # Create a dataframe to compare the models
        model_compare = pd.DataFrame(model_scores, index=['accuracy'])
        # Plot the model comparison
        model_compare.T.plot.bar(figsize=(10, 6))
        plt.xticks(rotation=0)
        plt.savefig('./figure/model_compare_plot.png')


        
        return model_compare
    



    
 




# main function:

if __name__ == "__main__":
    path = "./Dataset/"
    clf = DisasterTweetsClassification(path)
    df_train = clf.read_data(path + "train.csv")
    df_test = clf.read_data(path + "test.csv")
    df_submission = clf.read_data(path + "sample_submission.csv")

    print("Show the first 5 rows:\n")
    print(df_train.head())
    print("=====================================================================================================")

    print("Show the shape of the dataframes:\n")
    df_train.name = "Train"
    df_test.name = "Test"
    df_submission.name = "Submission"

    clf.show_shape(df_train)
    clf.show_shape(df_test)
    clf.show_shape(df_submission)
    print("=====================================================================================================")
    clf.show_info(df_train)
    print("=====================================================================================================")
    # show the missing values in the dataframes
    clf.show_missing_values(df_train)
    clf.show_missing_values(df_test)
    clf.show_missing_values(df_submission)
    print("=====================================================================================================")

    # plot the top 20 keywords
    clf.plot_top_keywords(df_train)

    # Replacing the ambigious locations name with Standard names
    df_train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},inplace=True)
    
    # plot the top 5 locations
    clf.plot_top_locations(df_train)

    # drop the columns
    columns = ['keyword', 'location'] # columns to be dropped because they are involves too many missing values. 
    df_train = clf.drop_columns(df_train, columns)
    df_test = clf.drop_columns(df_test, columns)

    # plot the countplot for the target variable
    clf.plot_disaster_count(df_train)

    # get X and y
    X , y = clf.get_X_y(df_train)

    # clean the text
    corpus = clf.clean_text(X)

    # vectorize the corpus
    X, df = clf.vectorize(corpus)
    print(df.head())    
    print("=====================================================================================================")

    # train test split
    X_train, X_test, y_train, y_test = clf.train_test_split(X, y)

    # train models
    model_compare = clf.train_models(X_train, X_test, y_train, y_test)
    print(model_compare)
    print("=====================================================================================================")

