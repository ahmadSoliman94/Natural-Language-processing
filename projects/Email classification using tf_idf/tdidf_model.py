# import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


######################## Email classification using tf_idf ########################

class model: 
    
    '''
    in this class we will create a model that will classify emails as spam or not spam using tf_idf vectorizer.
    '''
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, path):
        df = pd.read_csv(path,sep='\t',names=['Status','Message'])
        return df
    
    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def vectorize_data(self):
        train_vec = self.vectorizer.fit_transform(self.X_train).toarray()
        test_vec = self.vectorizer.transform(self.X_test).toarray()
        return train_vec, test_vec
    
    def train_model(self, X, y):
        self.model.fit(X, y)

    
    def predict(self, X):
        return self.model.predict(X)
    
    
    
    

if __name__ == '__main__':
    tf_idf_model = model()
    
    # 1. load data
    df = tf_idf_model.load_data('./dataset/sms.csv')
    print(df.head())
    print('#####################################################')
    print(len(df))
    
    print('#####################################################')
    # 2. convert ham and spam to 1 and 0
    df.loc[df["Status"]=='ham',"Status",]=1 # 1 for not spam
    df.loc[df["Status"]=='spam',"Status",]=0 # 0 for spam
    
    print(df.head())
    
    # 3. split data
    df_x = df['Message']
    df_y = df['Status']
    
    # 4. split data into train and test
    x_train, x_test, y_train, y_test = tf_idf_model.split_data(df_x, df_y)
    print('#####################################################')
    print(f"shape of x_train: {x_train.shape}\n" + " " + f"shape of y_train: {y_train.shape}\n" 
          + " " + f"shape of x_test: {x_test.shape}\n" + " " + f"shape of y_test: {y_test.shape}")
    
    # 5. vectorize data
    train_vec, test_vec = tf_idf_model.vectorize_data()
    print('#####################################################')
    print(train_vec[0])
    
    # 6. train model
    y_train = y_train.astype('int') # we need to convert y_train to int, because it is a string.
    tf_idf_model.train_model(train_vec, y_train)
    
    # 7. test model
    predictions = tf_idf_model.predict(test_vec)
    print('#####################################################')
    print(predictions[:20])
    
    y_test = np.array(y_test)
    print(y_test[:20])
    
    # 8. get accuracy
    count=0
    for i in range (len(predictions)):
        if predictions[i]==y_test[i]:
            count=count+1
    print('#####################################################')
    print("accuracy: ", count/len(predictions))