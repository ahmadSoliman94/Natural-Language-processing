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