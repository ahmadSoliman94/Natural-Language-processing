import gensim.downloader as api
import numpy as np


# ================================= Word2Vec ================================= # 

# Load the model
model = api.load("word2vec-google-news-300") # 300-dimensional word vectors trained on Google News corpus (3 million words and phrases)

# Get the vector of a word
# print(model['computer'])
# print(model['computer'].shape)

# Get the most similar words
w1 = model['King']
w2 = model['Queen']
print(np.linalg.norm(w1-w2)) # linear algebra norm of the difference between the two vectors, output: 2.4796925

print(model.most_similar(positive=['woman', 'king'], negative=['man'])) # here king - man + woman = queen , output : [('queen', 0.7118192315101624), ('monarch', 0.6189674139022827), ('princess', 0.5902431602478027), ('crown_prince', 0.5499460697174072), ('prince', 0.5377322435379028), ('kings', 0.5236846208572388), ('Queen_Consort', 0.5235946178436279), ('queens', 0.5181132555007935), ('sultan', 0.5098593235015869), ('monarchy', 0.5087411403656006)]


# not matching words
print(model.doesnt_match("house garage store sea".split())) # output: sea


# get the similarity between two words
print(model.similarty('iphone', 'samsung')) # output: 0.7064949

# get the most similar words
print(model.most_similar('cat')) # output: [('cats', 0.8099371194839478), ('dog', 0.7609455585479736), ('kitten', 0.7464982271194458), ('feline', 0.7324463129043579), ('beagle', 0.7156392331123352), ('puppy', 0.707610011100769), ('chihuahua', 0.7073332071304321), ('pup', 0.6910051107406616), ('golden_retriever', 0.6906319856643677), ('poodle', 0.6834224462509155)]


'''
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.

mathematically, it is defined as follows:
A.B = |A|.|B|.cos(theta) --> cos(theta) = A.B / |A|.|B| , where A.B is the dot product of A and B, |A| is the norm of A, |B| is the norm of B, theta is the angle between A and B.
norm of a vector is the length of the vector, mathematically, it is defined as follows: |A| = sqrt(A1^2 + A2^2 + ... + An^2) , where A1, A2, ..., An are the elements of the vector A.
for example:
A = [1, 2, 3]
B = [4, 5, 6]
A.B = 1*4 + 2*5 + 3*6 = 32
|A| = sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
|B| = sqrt(4^2 + 5^2 + 6^2) = sqrt(77)
cos(theta) = 32 / (sqrt(14) * sqrt(77)) = 0.9746318461970762
'''