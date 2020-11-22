# CD-APIR
Concept Drift-aware cloud service APIs Recommendation


To adapt users’ preference drifts and provide effective recommendation results to composite cloud system developers, we propose a concept drift-aware temporal cloud service APIs recommendation approach for composite cloud systems (or CD-APIR). CD-APIR track users temporal preferences through users’ behavior-aware information analysis. Singular Value Decomposition (SVD) and Jensen-Shannon (or JS) divergence are utilized to predict ratings in the user-service matrices. The steps are as follows: 

STEP 1 Algorithm preprocess the data by the code in “data_prepare”. The matrix of “old time window” and the matrix of “new time window” are also assigned.

STEP 2 Singular value decomposition of two matrices in new time window and old time window respectively. 

STEP 3 Calculate the JS divergence of each user based on the user’s ratings for the each service in the two window periods and predict the users’ ratings. 

STEP 4 Recommend cloud service APIs.


Experiments were implemented using Python 3.7.3 on win32.


We will continue to optimize CD-APIR.
