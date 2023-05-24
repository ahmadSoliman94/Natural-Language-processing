from sklearn.feature_extraction.text import CountVectorizer


# ==================== CountVectorizer ====================

'''
CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. 
'''

# initialize CountVectorizer
cv = CountVectorizer()

# example text
text = ["he likes likes to play fottball everyday ", "he and she are the best"]


# fit and transform the text
print(cv.fit_transform(text).toarray()) 

# get feature names
print(cv.get_feature_names_out())


# convert to dataframe
import pandas as pd
df = pd.DataFrame(cv.fit_transform(text).toarray(), columns=cv.get_feature_names_out())
print(df)