# Project motivation

## Problem statement

Machine Learning has been subjected to a wide array of applications. One such domain is the financial industry wherein the advent of modern technology has propelled new cases of fraud. The focus of this project is to detect credit card fraud using machine learning.
Credit card payment now covers as the primary option for making online transactions by the end user. However, no blessing is without its drawbacks. With the advent of modern technologies, the thieves have found new ways to circumvent the security to commit financial fraud. In 2020, the US accounted for 35.83% of the worldwide payment card fraud losses. Over the next 10 years, card industry losses to fraud will collectively amount to $408.50 billion.
Withstanding the above, this project intends to tackle the issue of credit card fraud with the aid of machine learning.

## Significance

Credit card fraud is a serious issue because it can cause significant financial losses for both individuals and financial institutions. Credit card fraud can also have broader economic consequences, as it can increase the cost of goods and services due to the added costs of fraud prevention and mitigation. Given the increasing amount of electronic transactions, the cost of fraudulent activities continues to rise. Making it important to have robust systems to detect and prevent it.
Machine learning has already provided extensive help in the financial industry such as when a user defaults. This maybe a lucrative opportunity to create a more efficient tool make predictions on new, unseen transactions, and flag any that it predicts to be fraudulent. Learning algorithms may be used to identify patterns or anomalies in the credit card transactions data that deviate from the normal behavior. These anomalies can then be flagged as potential fraud.

##  Methodology

This project already has the Kaggle dataset needed to jumpstart the process. A neural network will be the main tool employed for the purpose of solving this problem.

## Library and frameworks used

- Programing language: Python
- Library used: Pandas, Numpy, Sci-Kit learn, Pytorch, Imbalance learn, Matplotlib and seaborn

# Process

## Exploratory Data Analysis

The dataset being used throughout this project is the Credit Card Fraud Detection dataset from Kaggle. The dataset shown below contains 31 features; 28 of which have been PCA transformed and are labeled V1 through V28, Time which represents the number of seconds elapsed between the given transaction and the first transaction in the dataset, Amount which denotes the transaction amount, and finally the target feature Class which is a binary feature that represents whether or not a transaction is fraud (1) or not (0). The dataset is, however, extremely imbalanced as it contains 284,807 transactions and only 492 of those transactions are fraud. This means that in total the positive class makes up only 0.172% of all transactions. 

## Pre-processing 

The imbalanced nature of the dataset was counteracted by using Oversampling with SMOTE. SMOTE allowed us to effectively use the minority, in this case, positive class to its fullest potential by oversampling said class and undersampling the majority class. Then once we prepared SMOTE to balance our data we moved onto splitting the dataset into training and testing sets, ensuring to use Stratified-K-Fold to help split our data to ensure a useful amount of fraud samples are included in each fold.

## Approach

This project consists of using a multi-head attention transformer model in addition with SMOTE for data resampling, dropout regularization to help our model prevent overfitting, batch normalization which helps process the data for each layer during runtime to massively speed up performance. A more optimized model was used based on the previous model that uses both LSTM and GRU layer.
The multi-head attention transformer is called FraudModel, and the optimized model of the FraudModel shall be called FraudModel_LSTMGRU. Following that the data was tested with different ratios of fraud vs non-fraud transactions. 
The dataset was first resampled to be a 1:1 ratio with 492 fraud transactions. Then it was tested, trained, and evaluated the model with a 1:1 ratio with 500 transactions, followed by incremental jumps in fraud transactions by 500. So it was followed by 1:1 ratio with 1000 fraud, then 1:1 with 1500 fraud, up until 1:1 ratio with 5500 fraud transactions.
Once done it was tested with different test ratios of fraud to not fraud transactions ranging from 1:1 to 1:10 ratios 1:1 to 1:10 ratios.

## Algorithms used

- KNN: KNN is a type of instance-based machine learning algorithm uses distance to find the ‘K’ nearest neighbor instances to the unlabeled data point, then predict the class of the unlabeled data point given the proximity to other data points (taking their class into account).
- Gaussian Naive Bayes: Gaussian Naive Bayes in a probabilistic machine learning algorithm based on the extended Bayes Theorem. It is called naive as it is assumed that the features given are all independent from each other, however in the real-world the features can be correlated and the model has no mechanisms in place to counteract that fact.
- Decision Tree Classifier: Decision Tree Classifier is a tried and true tree based model that in our case uses gini impurity to calculate ideal feature splits to best divide the data into their respective classes. 
- Logistic Regression is a supervised machine learning linear model which is commonly known and used as a linear classifier. This is done by fitting the model to the data then using a decision boundary that minimizes the given loss function, in this case the loss function used was the L2 loss function which is the default. The parameters given were all of the default parameters, no optimization of the Logistic Regression model was done. 

# Results 

## Conclusion

The KNN model outperformed FraudModel & FraudModel_lstmgru in terms of accuracy the **FraudModel_lstmgru** consistently ranked **#1** or **#2** in all metrics across **ALL ratio/sample-size splits**.  **SMOTE** successfully **oversampled** our **minority positive class** while **undersampling** our **majority negative class** and allowed **our model** to perform **better** than otherwise possible. The **Transformer with Multi-Head Attention** was able to achieve amazing results across many metrics.

# Future Scope

For future work it has been proposed that we can attempt to test this model on ratios and splits beyond what was originally covered in the scope of this project. While the current 18 hour runtime for the Python file is discouraging to perform more extensive testing
