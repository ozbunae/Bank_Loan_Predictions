# Project_3
This is a bank loan model created for my third project at flatiron.

## Project Description:


### 1.1 The data:
|Columns|Descriptions|Type|
|:------|:-----------:|:----|
|ID|Customer ID|Continuous|
|Age|Customer's age in completed years|Continuous|
|Experience|years of professional experience|Continuous|
|Income|Annual income of the customer (thousands)|Continuous|
|ZIPCode|Home Address ZIP code.|Categorical|
|Family|Family size of the customer|Categorical|
|CCAvg|Avg. spending on credit cards per month (thousands)|Continuous|
|Education|Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional|Categorical|
|Mortgage|Value of house mortgage if any. (thousands)|Continuous|
|Personal Loan|Did this customer accept the personal loan offered in the last campaign?|0 = No 1= Yes|
|Securities Account|Does the customer have a securities account with the bank?|0 = No 1= Yes|
|CD Account|Does the customer have a certificate of deposit (CD) account with the bank?|0 = No 1= Yes|
|Online|Does the customer use internet banking facilities?|0 = No 1= Yes|
|CreditCard|Does the customer use a credit card issued by UniversalBank?|0 = No 1= Yes|


First looking at the head of this data and the description of basics like quartiles, mean, and standard deviation, we see that it is a really clean data set. There are no missing values. The minimum and maximum do not stray from the interquartile range too dramatically. We have 5000 entries which is decent. For machine learning it would be preferable for us to have more than this for the model to learn from. The cleanliness of this data set lends itself well, however.

Taking a look below at the histograms for each column or 'feature', it can be observed that our continuous variables include Age, Experience, Income, CCAvg, and Mortgage. Both CCAvg and Mortgage indicate two things:

The first is many of the rows are ticked with 0. All of those rows indicate people who do not have a Credit Card or Mortgage, respectively.

The second is the remaining values representing the amount of average credit card expenditure per month and mortgage amount in thousands, respectively.

These seperate issues will be handled in the data cleaning.

The categorical variables include Family and Education.

Finally we have a seies of categorical variables that were yes or no questions represented by 1s and 0s. These include Securities Account, CD Account, Online, and Credit Card.

The final column, Personal Loan, will act as our 'y' variable. The variable we are trying to predict in the future. The important thing to remember about this data set and this project is that Personal loan does NOT measure whether or not someone was approved for a Personal Loan or not. The subjects in this study have already been approved for a Personal Loan and it was recorded if they accepted or not. We are trying to measure the likeliehood that someone will accept the loan offer from this bank.

ID and Zipcode will both be dropped. All of the data essentially comes from the same zipcode so geographic information is irrelavent. If we want to expand the market nationally or internationally then geographic features may become important.

### 1.2 Cleaning the Data
This dataset was relatively easy to work with.  It was very complete and did not have any strange data types or objects.

Mortgage and credit card average need to be split by T/F and continuous data.
Create two new columns:
* The first is replacing any numerical value not 0 with 1.
* The second is using what numerical value there is to create a continuous set of data for Mortage amount and average monthly CC spending

Both the z score and intequantile methods were far too aggressive. Both methods cut the data in half. Removing outliers also killed the Personal_Loan column which is essential to this project.
This was a very clean data set so it is possible they were already removed. However there is the problem of the skew...

The final method that was chosen was to select individual columns, or features, and decide on a reasonable range in which to keep them.  For example on mortgage, although it is heavily skewed to the right, the data is more complete that removing outliers and will not affect our model in a positive or negative way.

### 1.3 The Business Problem

The business problem is to expand the number of 'asset customers' the bank has. This means customers who are paying interst to the bank and there for increasing the bank's overall networth. The idea is to create a machine learning model that will accurately predict the liklihood someone will accept a personal loan. Using this informatino the bank can selectively market their personal loan program and expand their number of 'asset customers'.

## Exploratory Data Analysis:

### 2.1 
In this exploratory data analysis I try to explore the obvious relationships.  We assume that the higher someone's income is the higher their mortgage will be.  Digging deeper we assume that more years of experience the higher their income will be.  Finally the more augmented all these factors are, the safer it is to assume that there is a higher monthly credit card bill as well.  We rely on the assumption that all of these factors have specific linear relationships with one another.  

What we want to explore however, is how ALL of these variables affect our final model.

## Building Models:

### 3.1 Logistic Regression
Unlike Linear Regression Logistic Regression accounts for Categorical variables.

Logistic regression is easier to implement, interpret, and very efficient to train. If the number of observations is lesser than the number of features, Logistic Regression should not be used, otherwise, it may lead to overfitting. It makes no assumptions about distributions of classes in feature space.

### 3.2 Decision Tree

Trees answer sequential questions which send us down a certain route of the tree given the answer. The model behaves with “if this than that” conditions ultimately yielding a specific result.

### 3.3 Random Forest

From an article on Towards Data Science:

One way Random Forests reduce variance is by training on different samples of the data. A second way is by using a random subset of features. This means if we have 30 features, random forests will only use a certain number of those features in each model, say five. Unfortunately, we have omitted 25 features that could be useful. But as stated, a random forest is a collection of decision trees. Thus, in each tree we can utilize five random features. If we use many trees in our forest, eventually many or all of our features will have been included. This inclusion of many features will help limit our error due to bias and error due to variance. If features weren’t chosen randomly, base trees in our forest could become highly correlated. This is because a few features could be particularly predictive and thus, the same features would be chosen in many of the base trees. If many of these trees included the same features we would not be combating error due to variance. With that said, random forests are a strong modeling technique and much more robust than a single decision tree. They aggregate many decision trees to limit overfitting as well as error due to bias and therefore yield useful results.

### 3.4 K-Nearest Neighbor

KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).

### 3.5 Support Vector Machine



### 3.6 XG Boost

## What helps us decide which model is best?



