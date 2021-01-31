# Concept Drift-aware cloud service APIs Recommendation (CD-APIR)

To adapt users’ preference drifts and provide effective recommendation results to composite cloud system developers, we propose CD-APIR. CD-APIR tracks users’ temporal preferences through users’ behavior-aware information analysis. Singular Value Decomposition (SVD) and Jensen-Shannon (or JS) divergence are utilized to predict ratings in the user-service matrices. The steps are as follows:

STEP 1: Recommendation/Data_preparation

Preprocess the data. The matrix of the “old time window” and the matrix of “new time window” are also preprocessed.

STEP 2: Recommendation/SVD.py

Singular value decomposition of two matrices in new time window and old time window respectively.

STEP 3: Recommendation/JS divergence and prediction.py

Calculate the JS divergence of each user based on the user’s ratings to each service in the two window periods.

STEP 4: Recommendation/Recommendation.py

Recommend cloud service APIs based on preference drift detection.

Experiments were implemented using Python 3.7.3 on win32.


***Please refer to the following paper for a detailed description of the CD-APIR approach:

[1] Lei Wang, Yunqiu Zhang, and Xiaohu Zhu, “Concept Drift-Aware Temporal Cloud Service APIs Recommendation for Building Composite Cloud Systems”, Journal of Systems and Software, 2021, 174:110902. DOI: 10.1016/j.jss.2020.110902.

***IF YOU ARE INSPIRED BY THIS SOURCE CODE IN PUBLISHED RESEARCH, PLEASE CITE THE ABOVE PAPER. THANKS!

***IF YOU ARE INSPIRED BY OUR TECHNIQUE FOR YOUR COMMERCIAL USED SYSTEMS, COULD PLEASE LET US KNOW? THANKS!

E-MAIL: leiwangchn@163.com.
