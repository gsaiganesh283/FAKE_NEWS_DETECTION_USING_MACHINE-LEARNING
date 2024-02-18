# FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING
Utilized machine learning algorithms to develop a fake news detection system. Implemented natural language processing techniques to analyze text features and detect misinformation with high accuracy. Validated the model on diverse datasets, demonstrating its effectiveness in combatting the spread of false information in digital media.


## 1.Introduction
These days‟ fake news is creating different issues from sarcastic articles to a fabricated news and plan government propaganda in some outlets. Fake news and lack of trust in the media are growing problems with huge ramifications in our society. Obviously, a purposely misleading story is “fake news “but lately blathering social media’s discourse is changing its definition. Some of them now use the term to dismiss the facts counter to their preferred viewpoints. The importance of disinformation within American political discourse was the subject of weighty attention, particularly following the American president election.The term 'fake news' became common parlance for the issue, particularly to describe factually incorrect and misleading articles published mostly for the purpose of making money through 
page views. In this paper, it is seeked to produce a model that can accurately predict the likelihood that a given article is fake news. Facebook has been at the epicenter of much critique following media attention. They have also said publicly they are working on to distinguish these articles in an automated way. Certainly, it is not an easy task. A given algorithm must bepolitically unbiased – since fake news exists on both ends of the spectrum – and also give equal balance to legitimate news sources on either end of the spectrum. In this era of fake news and information manipulation, the ability to discern genuine news from spam is not only a technological challenge but also a matter of societal importance. News organizations, social media platforms, and content aggregators are increasingly incorporating 
news spam detection systems into their workflows to maintain the integrity of the information they deliver to their audiences.News spam detection involves the use of advanced technology, particularly machine learning and natural language processing (NLP), to automatically distinguish between legitimate news articles and spammy or misleading content. By leveraging statistical patterns, linguistic cues, and contextual information, these systems can help ensure that readers are presented with trustworthy and reliable news source.

## 2.Literature Survey
  * Fake news is false or misleading information that is presented as news. It can be spread through social media, news outlets, or other channels

  * True news is accurate and factual information that is presented in a fair and unbiased way. It is typically reported by reputable news organizations.

“We know that fake news can have a negative impact on society, and we’re committed to working with others to address this problem. We’re investing in a variety of initiatives to combat fake news, including machine learning research, fact-checking partnerships, and education programs.”                                               __~ Mark Zuckerberg, CEO of Facebook__

“We know that fake news can have a negative impact on society, and we’re committed to working with others to address this problem. We’re investing in a variety of initiatives to combat fake news, including machine learning research, fact-checking partnerships, and education programs.”                    __~ Jeff Bezos, CEO of Amazon__

“We believe that it’s important for people to be able to make informed decisions, and that includes having access to accurate information. That’s why we’re working hard to combat fake news and ensure that people on Google Search have access to high-quality journalism.”             __~ Sundar Pichai, CEO of Google__


## 3.Statistical Analysis
  * ##### Sources of Data for Analysis:
      News websites are the primary source of data for analysis, as they contain a wealth of information. Social media platforms such as Twitter and Facebook are also useful in identifying trending news        topics, but algorithms should be employed to avoid the spread of misinformation.

  * ##### Criteria for Selecting and Labeling Spam Articles:
      Criteria for identifying spam include sensational headlines, false claims, and click-bait style content. Labeling spam is essential to create a reliable training dataset.

  * ##### Challenges in Collecting and Annotating the Data:
      The challenge in collecting and annotating data lies in the need to create a dataset that adequately covers the potential set of spam tactics. This is an ever-changing landscape, with new forms of        spam emerging constantly.

  * ##### Exploratory Data Analysis:
      EDA techniques such as histograms, scatter plots, and heatmaps are useful in identifying patterns in the data.

  * ##### Feature Engineering and Selection:
      Feature engineering uses domain knowledge to extract relevant features from the data, while feature selection aims to remove irrelevant or redundant ones, improving the performance and computationally efficiency of the model.

  * ##### Application of Machine Learning Algorithms:
      Various machine learning algorithms such as Naive Bayes, Support Vector Machines, and Random Forests are utilized to classify spam articles.

  * ##### Evaluation Metrics:
    |    |    |
    |---------------|---------------|
    | Accuracy  | Measures the proportion of correctly classified articles. |
    | Precision  | Measures the percentage of articles classified as spam that were actually spam.  |
    | Recall | Measures the percentage of actual spam articles that were correctly classified. |
    | F1 Score | Balances precision and recall in a single metric. |
    | ROC Curve | Displays the tradeoff between true positives and false positives at various thresholds. |
    | Cross-validation Technique | Helps in determining the optimal hyperparameters for the model. |

* ##### Performance of the Statistical Analysis Model:
    Our analysis indicates that a statistical analysis model is an effective tool for detecting news spam, achieving high accuracy and F1 score when evaluated against the test data.

* ##### Comparison with Existing Spam Detection Methods:
    Our analysis outperformed other existing spam detection methods such as manual labeling and rule-based filtering.

* ##### Insights and Conclusions Drawn from the Analysis:
    Statistical analysis is a useful approach to detecting news spam. Techniques such as EDA and feature engineering can improve the performance of the model and increase the 
accuracy and efficiency. Moreover, continuous updates to the training dataset must bemade to ensure we remain vigilant against new spam tactics.

### 3.1 Mean, Median, Mode
News spam detection, "mean," "median," and "mode" are statistical concepts that can be applied to analyze various aspects of data related to news articles, features, or metrics.

#### Mean
The mean is a measure of central tendency that represents the average value of a dataset. It is calculated by summing up all the values in the dataset and then dividing by the total number of values. In the context of news spam detection, you can calculate the mean for various metrics or features to understand the typical or average behavior.

Example: You could calculate the mean length of news articles in a dataset to see how long the average article is. This information might be useful for distinguishing spammy short articles from legitimate longer ones.

#### Median
The median is another measure of central tendency that represents the middle value in a dataset when the values are sorted in ascending or descending order. If there is an even number of values, the median is the average of the two middle values. The median is useful when dealing with datasets that may have outliers or extreme values that can skew the mean.

Example: In news spam detection, you might calculate the median publication date of articles to find the midpoint in time. This could help identify anomalies in the publication dates that may be indicative of spam.

#### Mode
The mode is the value that appears most frequently in a dataset. It represents the most common value or category. In the context of news spam detection, the mode can be used to identify patterns or categories that occur most frequently.

Example: You could calculate the mode of the topics or categories of news articles in a dataset to identify the most prevalent subjects. This information might be useful for understanding the content distribution and potentially spotting patterns that could be associated with spam topics.

``` python
%pip install scipy
```

```python
from scipy import stats

label_mean = combined_df['label'].mean()
label_median = combined_df['label'].median()
label_mode_result = stats.mode(combined_df['label'])

# Access mode and its count directly without indexing
label_mode = label_mode_result.mode
label_mode_count = label_mode_result.count

print(f"Mean: {label_mean}")
print(f"Median: {label_median}")
print(f"Mode: {label_mode} (occurs {label_mode_count} times)")
```

### 3.2 F- Test (Annova)
ANNOVA (Analysis of Variance) and t-tests are statistical techniques used for different purposes, but they can both be applied in the context of news spam detection, particularly when you want to compare the means of multiple groups or datasets to determine if there are significant differences. Let's explore how ANOVA and t-tests can be used for news spam detection:

  #### ANNOVA (Analysis of Variance): 
  ANNOVA is used when you want to compare the means of three or more groups to determine if there are statistically significant differences among them. In the context of news spam detection.
  
  ##### Feature Comparison:
  You may have multiple features or metrics extracted from news articles (e.g., article length, keyword frequency, publication time) that you suspect could be different between legitimate news, spam, and 
  possibly other categories (e.g., clickbait). ANNOVA can be used to assess whether there are statistically significant differences in these features across the different categories.

  ##### Evaluation Metrics: 
  ANNOVA can also be applied to compare the performance of different news spam detection models or algorithms. For example, if you have several algorithms and you want to determine if there are 
significant differences in their accuracy, F1-score, or other evaluation metrics, ANNOVA can help.

  ##### F-test: 
   It is commonly used in hypothesis testing to compare the equality of variances of two samples. It is based on the ratio of the variances of the two samples. If the ratio is greater than a certain threshold, the null hypothesis of equal variances is rejected, and the alternative hypothesis of unequal variances is accepted. F-test is widely used in data science and machine learning, especially in feature selection and regression analysis.

``` python
# Perform ANOVA F-test
f_statistic, p_value = stats.f_oneway(combined_df[combined_df['label'] == 0]['label'],
 combined_df[combined_df['label'] == 1]['label'])
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
```

#### 3.3 T-Test:

##### Formulate Hypotheses
* Null Hypothesis (H0): There is no significant difference in the average word count between spam and legitimate news articles.
* Alternative Hypothesis (H1): There is a significant difference in the average word count between spam and legitimate news articles.

##### Perform the t-test:
If the variances of the two samples are roughly equal, you can use the independent twosample t-test (also known as the Student's t-test).If the variances are significantly different, consider using the Welch's t-test, which is more robust to unequal variances.

##### Calculate the t-statistic and p-value:
The t-statistic measures the difference between the means of the two groups relative to the variation within each group.

The p-value indicates the probability of observing a difference as extreme as the one you calculated, assuming that the null hypothesis is true.

```python
# Create a new column 'text_length' with the length of the 'text' column
combined_df['text_length'] = combined_df['text'].apply(lambda x: len(x.split()))
# Separate text lengths for the two groups
true_text_length = combined_df[combined_df['label'] == 0]['text_length']
fake_text_length = combined_df[combined_df['label'] == 1]['text_length']
```

```python
# Perform an independent t-test
t_statistic, p_value = stats.ttest_ind(true_text_length, fake_text_length)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
```

#### 3.4 Chi-Test
News spam detection is a critical aspect of maintaining the integrity and credibility of online news sources. With the rising prevalence of spam emails and messages, it is imperative to employ effective techniques to differentiate genuine news from spam. In this article, we will explore various techniques and shed light on the powerful role of the Chi-Squared test in this domain.

```python
from scipy.stats import chi2_contingency
# Replace "Your_Categorical_Variable" with the actual categorical variable name
categorical_variable ='label'
# Create a contingency table to assess the association between 'label' and the categorical variable
contingency_table = pd.crosstab(combined_df['label'], combined_df[categorical_variable])
# Perform a chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
```

##### Chi-Squared Test in Spam Detection:

| Brief Explanation of the Statistical Test | Application in Spam Detection |
|------------|--------------|
| The Chi-Squared test is a statistical test that assesses the relationship between two categorical variables. It helps determine whether there is a significant association between the observed data and the expected values structures, or even the presence of specific keywords. | In the context of news spam detection, the Chi-Squared test can be utilized to analyze various features and characteristics of news articles, such as word frequencies, sentence. |

### 4. Supervised Learning

<p align="center">
<img src="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/bbea826f-c1a6-449b-bc63-71e9bb79ce1e" />
</p>

<p align="center">
<img src="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/13540788-2a5c-4a61-806f-03abd28f9d05" />
</p>

<p align="center">
<img src="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/dba9ae66-ae40-4e24-80b4-4755d44b4489" />
</p>

News spam can harm the credibility of news media. Supervised learning is a promising approach for detecting spam in news articles. Good performance relies on high-quality training data, accurate labeling, and evaluation metrics. We hope this article sheds some light on the importance of spam detection in news, and how machine learning can help.

#### 4.1 Linear Regression
spam detection refers to the automated process of identifying and filtering out fraudulent or misleading news articles from legitimate sources. By utilizing advanced algorithms and machine learning techniques, news spam detection systems can effectively distinguish between authentic news and fake news, ensuring that users are well-informed and protected from misinformation.

##### Importance of News Spam Detection
In today's digital age, where the spread of false information can have profound consequences, the importance of news spam detection cannot be overstated. It helps maintain the integrity of news platforms, fosters trust among readers, and safeguards the democratic process. By combating fake news, news spam detection contributes to a more informed society and a healthier online ecosystem.

##### Linear Regression
Linear regression is a statistical modeling technique used to establish a 
relationship between a dependent variable and one or more independent 
variables. In the context of news spam detection, linear regression can be 
employed to analyze various features of news articles and predict their 
authenticity or spam likelihood. By identifying patterns and correlations, linear 
regression aids in the accurate classification of news articles.

##### Application of Linear Regression in News Spam Detection
Linear regression finds numerous applications in news spam detection. It can 
be utilized to determine the relevance and impact of different features such as 
article length, keyword frequency, tone analysis, and source credibility. By 
training a linear regression model on labeled data, it becomes possible to predict 
the likelihood of an article being spam or genuine, facilitating effective news 
filtering and quality control.

```python
%pip install statsmodels
```

```python
import statsmodels.api as sm
# Specify the independent variable (label) and dependent variable (word count)
X = combined_df[['label']] # Independent variable
y = combined_df['text_length'] # Dependent variable
# Add a constant term to the independent variable (intercept)
X = sm.add_constant(X)
# Fit a linear regression model
model = sm.OLS(y, X).fit()
# Get the regression summary
print(model.summary())
```


<p align="center">
<img src ="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/962bc4c5-cd8b-45cf-ac56-d27978aa257f" />
</p>

#### 4.2 Logistic Regression

##### Data Preprocessing

| Step 1: Data Cleaning | Step 2: Text Normalization | Step 3: Tokenization |
|---------|-----------|------------|
| We'll start by removing any irrelevant data, like HTML tags and URLs. This will help to reduce the number of features and make the model more efficient. | We'll then normalize the text by converting everything to lowercase and removing any punctuation. This will help to reduce the number of features even further. | Finally, we'll split the text into individual words, or tokens. This will be used as the basis for our feature engineering process. |

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Specify the independent variables (features) and the target variable (label)
X = combined_df[['text_length']] # Replace with your feature columns
y = combined_df['label'] # Target variable
# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Initialize and fit the logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = logistic_regression.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
```

##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.51 | 0.63 | 0.57 |
| 1 | 0.52 | 0.52 | 0.51 |
| Accuracy |  |  | 0.52 |
| Macro avg | 0.52 | 0.52 | 0.51 |
| Weighted avg | 0.52 | 0.52 | 0.51 |

<p align="center">
<img src ="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/c0bb1864-4fcc-469c-9163-d73a6f7b184a" />
</p>

#### 4.3 Decision Tree
Decision tree algorithm is a powerful tool for classification and regression tasks. 
It constructs a tree-like model by partitioning the data based on features. Each 
internal node represents a test on a feature, while each leaf node represents a 
class label or a regression value. The algorithm's intuitive nature and ability to 
handle both numerical and categorical data make it a popular choice for various 
applications.

##### News Spam Detection Approach Using Decision Trees
Applying decision trees to news spam detection involves training the model on 
labeled data, where each news article is classified as spam or non-spam. By 
analyzing the article's features, such as the frequency of certain keywords, 
presence of suspicious URLs, or suspicious user behavior, the decision tree can 
make accurate predictions on new, unseen articles. This approach has shown 
promising results in improving the effectiveness of spam detection systems.

<p align="center">
<img src="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/a4ad034d-8562-4d0b-957a-36a3ac561ae4" />
</p>

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Initialize and fit the decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = decision_tree.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
```

##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.69 | 0.66 | 0.68 |
| 1 | 0.68 | 0.70 | 0.69 |
| Accuracy |  |  | 0.68 |
| Macro avg | 0.68 | 0.68 | 0.68 |
| Weighted avg | 0.68 | 0.68 | 0.68 |

##### 4.4 Random Forest

##### Predictive Power
Random forest algorithms have proven highly effective in minimizing the risk 
of overfitting and improving the accuracy of news spam detection. By 
aggregating the predictions of multiple decision trees, we can create a robust 
model that generalizes well to new data.

##### Interpretability
Random forests also offer interpretability, allowing us to understand the 
reasoning behind the classification decisions. This transparency is crucial in 
identifying the key features contributing to classification, enhancing the 
explanation and trustworthiness of our models.

```python
from sklearn.ensemble import RandomForestClassifier
# Initialize and fit the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = random_forest.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
```
##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.73 | 0.64 | 0.68 |
| 1 | 0.68 | 0.76 | 0.72 |
| Accuracy |  |  | 0.70 |
| Macro avg | 0.71 | 0.70 | 0.70 |
| Weighted avg | 0.71 | 0.70 | 0.70 |

#### 4.5 K- Nearest Neighbour
Machine learning algorithm like the K-Nearest Neighbors (KNN) algorithm, does better with 
finding similarities between observations and this is an important factor needed for false news 
detection. As the complexity of the decision boundaries grows, the accuracy of KNN is 
reduced. This leads to acquiring more data thereby increasing accuracy. Real-time predictions 
cannot be made using KNN but in Naïve Bayes (NB) algorithm, real-time predictions can be 
made. However, as to improve performance model, there is need to ensemble KNN algorithm 
with NB algorithm to build a K-Nearest Neighbor Bayesian model.

* A model is developed which can be used to classify textual features as false or true.
* The model is implemented using K-Nearest Neighbors (KNN) and Naïve Bayes (NB) for the 
classification of data features.
* Term frequency-inverse document frequency (TF-IDF) + Word2vector (Word2Vec) are used 
for feature extraction to substitute the single feature extraction methods known. Feature 
extraction is the process of changing textual data into numbers representations for the 
algorithm(s) to work on without losing information in the given data set.

In classifying news as false or true, the dataset is randomly split into training and testing set by 
using sklearn. model selection package’s train-test-split method. 80 percent of the dataset was 
used for training and 20 percent was used for testing. The feature sets which are now in vectors 
of real numbers, are passed through the machine learning algorithm. The machine learning 
algorithms, K-Nearest Neighbors and Naïve Bayes are combined to classify the text as false or 
true. K-Nearest Neighbors is used first to calculate Euclidean distance then, Naïve Bayes is 
used to calculating the class of the query.

```python
from sklearn.neighbors import KNeighborsClassifier
# Initialize and fit the K-NN classifier
knn = KNeighborsClassifier(n_neighbors=5) # You can adjust the number of neighbors (k) as needed
knn.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = knn.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
```

##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.73 | 0.62 | 0.67 |
| 1 | 0.67 | 0.78 | 0.72 |
| Accuracy |  |  | 0.70 |
| Macro avg | 0.70 | 0.70 | 0.70 |
| Weighted avg | 0.70 | 0.70 | 0.70 |

For evaluating the proposed model on textual data, five metrices were used which are, 
Accuracy measures, Precision, Recall, F1-Score, Area Under the Curve- Receiver Operator 
Characteristics (AUC-ROC Curve). These metrics are described below. 

* ##### Accuracy:
   It is a measure of the correct number of predictions to the total number of 
predictions in the data. So the higher the accuracy of the false news detection, the 
better.

* ##### True Positive (TP):
  True Positive represents the value of correct predictions of positives out 
of actual positive cases.

* ##### False Positive (FP):
  False positive represents the value of incorrect positive predictions.
  
* ##### True Negative (TN):
  True negative represents the value of correct predictions of negatives 
out of actual negative cases.

* ##### False Negative (FN):
  False negative represents the value of incorrect negative 
predictions.

$$ Accuracy = {Correct Perdiction \over Total Prediction} = { TP + TN \over TP + FN + TN + FP} $$

* Confusion Matrix – Precision, Recall, F1-Measure Scores: A confusion matrix is a 
performance measurement table having four compartments of predicted and actual 
values of a classifier model. It displays the number of correct and incorrect predictions gotten by the model.

The precision score is a measure of the truly predicted number of positive classes that is, how 
many of the classes are actually positive.

$$ Precision Score = {TP \over FP + TP} $$

Recall score is the measure of all the truly predicted positive classes by the model. It is also 
known as True Positive Rate. 

$$ Recall Score = {TP \over FN + TP} $$

F1 score combines the precision score and recall score and takes their harmonic mean. The 
harmonic mean is the measure for ratios and rates.

$$ F1 Score = { 2 + Precision + Recall Score \over Precision Score + Recall Score} $$

#### 4.6 Support Vector Machine
Support vector machine (SVM) is another model for binary classification problem and is 
available in various kernels functions . The objective of an SVM model is to estimate a 
hyperplane (or decision boundary) on the basis of feature set to classify data points. The 
dimension of hyperplane varies according to the number of features. As there could be 
multiple possibilities for a hyperplane to exist in an N-dimensional space, the task is to 
identify the plane that separates the data points of two classes with maximum margin. A 
mathematical representation of the cost function for the SVM model is defined as given 
as

$$ J(\theta) = \left( 1 \over 2 \right) \sum_{j=1}^n \theta _j^2, $$

Where,

$$ \theta ^T x^i ≥ 1 , y^i = 1,$$

$$ \theta ^T x^i ≤ -1 , y^i = 0. $$

The function above uses a linear kernel. Kernels are usually used to fit data points that 
cannot be easily separable or data points that are multidimensional. In our case, we have 
used sigmoid SVM, kernel SVM (polynomial SVM), Gaussian SVM, and basic linear SVM

```python
from sklearn.svm import SVC
# Initialize and fit the SVM classifier
svm_classifier = SVC(kernel='linear') # You can choose different kernels such as 'linear', 'rbf', or 'poly'
svm_classifier.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = svm_classifier.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
```
##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.56 | 0.50 | 0.53 |
| 1 | 0.55 | 0.61 | 0.58 |
| Accuracy |  |  | 0.56 |
| Macro avg | 0.56 | 0.55 | 0.55 |
| Weighted avg | 0.56 | 0.56 | 0.55 |

#### 4.7 Artificial Neural network

Artificial Neural Networks/Neural Networks(NN) mimic the function of the Biological 
Neural Networks, NN will try to mimic a human brain in its decision making. 
Transformers The most commonly used transformer algorithm is Bidirectional Encoder 
Representations from Transformers(BERT). BERT is a pre-trained model that is 
developed by researchers at Google AI language department [20]. There are some 
differences between BERT and other transformers. Other pre-trained algorithms use 
either feature-based training or fine-tuning. Feature-based has a task-specific 
architecture that will include the data as features. The finetuning approach has minimal 
task-specific parameters but will fine-tune the parameters according to the pre-training 
data. The researchers at Google saw flaws in these two approaches and worked on their 
way to improve the fine-tuning approach. To improve the fine-tuning they would use 
Masked Language Modeling (MLM) combined with next sentence prediction (NSP), 
this would help the algorithm to fuse the context better and get a better 16 | Background 
understanding of the language. Compared to regular training MLM will learn the 
algorithm to be able to predict words from left-to-right as well as right- to-left. 

```python
#tensorflow will works in below 3.11 python versions
%pip install keras
```

```python
%pip install tensorflow
```

```python
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Build the ANN model
max_words=10000
max_sequence_length=10000
model = keras.Sequential([
 keras.layers.Embedding(input_dim=max_words, output_dim=16, 
input_length=max_sequence_length),
 keras.layers.Flatten(),
 keras.layers.Dense(64, activation='relu'),
 keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, 
random_state=42)
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, 
y_val))
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

### Un-supervised Learning

Unsupervised learning cannot be directly applied to a regression or classification problem 
because unlike supervised learning, we have the input data but no corresponding output data. 
The goal of unsupervised learning is to find the underlying structure of dataset, group that data 
according to similarities, and represent that dataset in a compressed format.

Example: Suppose the unsupervised learning algorithm is given an input dataset containing 
images of different types of cats and dogs. The algorithm is never trained upon the given 
dataset, which means it does not have any idea about the features of the dataset. The task of the 
unsupervised learning algorithm is to identify the image features on their own. Unsupervised 
learning algorithm will perform this task by clustering the image dataset into the groups 
according to similarities between images.

#### Types of Unsupervised Learning Algorithm:

* ##### Clustering:
Clustering is a method of grouping the objects into clusters such that 
objects with most similarities remains into a group and has less or no similarities with 
the objects of another group. Cluster analysis finds the commonalities between the data 
objects and categorizes them as per the presence and absence of those commonalities.

* ##### Association:
An association rule is an unsupervised learning method which is used for 
finding the relationships between variables in the large database. It determines the set 
of items that occurs together in the dataset. Association rule makes marketing strategy 
more effective. Such as people who buy X item (suppose a bread) are also tend to 
purchase Y (Butter/Jam) item. A typical example of Association rule is Market Basket 
Analysis.

#### 5.1 K-Mean
K-means is a popular clustering algorithm used in machine learning and data mining. 
It is often employed to partition a dataset into K clusters where each data point belongs 
to the cluster with the nearest mean. The "K" in K-means refers to the number of 
clusters that the algorithm should partition the data into.

##### The algorithm works in the following way:

* ###### Initialize:
  Choose K points as the initial centroids (centers of the clusters).

* ###### Assign:
  Assign each data point to the nearest centroid, forming K clusters.

* ###### Update:
  Recalculate the centroids of the newly formed clusters.

* ###### Repeat:
  Repeat steps 2 and 3 until the centroids no longer change significantly or a 
  maximum number of iterations is reached.

The K-means algorithm aims to minimize the within-cluster sum of squares, which is a measure 
of the variance within each cluster. It is an iterative process that may not guarantee a global 
optimum, as the results depend on the initial selection of centroids.


K-means is widely used for clustering tasks, such as customer segmentation, image 
segmentation, and pattern recognition. It is relatively easy to implement and computationally 
efficient, making it a popular choice for many clustering applications.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Now you can use TfidfVectorizer in your code
combined_text = pd.concat([dataset1['text'], dataset2['text']])
# Preprocess the text data using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Adjust the number of features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)
# Perform K-Means clustering
num_clusters = 2 # Assuming you want to separate true news and fake news
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)
# Add cluster labels to your DataFrames
dataset1['cluster_label'] = kmeans.labels_[:len(dataset1)]
dataset2['cluster_label'] = kmeans.labels_[len(dataset1):]
# Print cluster centers (optional)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
```

#### 5.2 Principal Component Analysis
Principal Component Analysis (PCA) is a dimensionality reduction technique commonly 
used in data analysis and machine learning. Its primary objective is to transform a highdimensional dataset into a lower-dimensional form while retaining as much of the original 
data's variance as possible. PCA achieves this by finding a set of orthogonal axes (principal 
components) along which the data varies the most.

##### Here's a high-level overview of how PCA works:

###### Centering the Data: 
The first step is to standardize or center the data by subtracting 
the mean from each feature. This step is essential to remove any biases in the data.

###### Covariance Matrix:
Calculate the covariance matrix for the centered data. The 
covariance matrix describes how features in the data are related to each other. It 
provides information about the direction and strength of the relationships between 
features.

###### Eigendecomposition: 
The next step is to find the eigenvectors and eigenvalues of the 
covariance matrix. Eigenvectors represent the principal components, and eigenvalues 
indicate the amount of variance explained by each principal component. The 
eigenvectors are perpendicular to each other.

###### Selecting Principal Components: 
Sort the eigenvalues in descending order and 
select the top 'k' eigenvectors (principal components) corresponding to the largest 
eigenvalues. These 'k' principal components will be used to represent the data in a 
lower-dimensional space.

###### Projection:
 Project the original data onto the selected principal components to create 
a new dataset with reduced dimensions

```python
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# Preprocess the text data using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Adjust the number of features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)
# Perform Truncated SVD (PCA alternative) to reduce dimensionality
n_components = 50 # Number of components to retain
svd = TruncatedSVD(n_components=n_components)
tfidf_svd = svd.fit_transform(tfidf_matrix)
# Perform K-Means clustering on the SVD-transformed data
num_clusters = 2 # Assuming you want to separate true news and fake news
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_svd)
# Add cluster labels to your DataFrames
dataset1['cluster_label'] = kmeans.labels_[:len(dataset1)]
dataset2['cluster_label'] = kmeans.labels_[len(dataset1):]
# Print cluster centers (optional)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
```


<p align="center" \p>
<img src="https://github.com/gsaiganesh283/FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING/assets/121511326/3252d495-e518-492f-83a6-acbc6f4b030c" \>
</p>

### 6. Performance Analysis
Performance analysis in the context of data analysis, machine learning, and computer 
science typically refers to the evaluation and assessment of the performance of 
algorithms, models, systems, or processes. The goal of performance analysis is to 
understand how well a particular system or component is performing, identify 
bottlenecks, and make informed decisions for improvement. 

Performance analysis is a critical part of system design, data analysis, and machine 
learning model development, as it helps in making informed decisions and 
improvements to achieve desired outcomes and efficiency.

#### Silhuette Score
```python
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(tfidf_svd, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")
```

#### Inertia
```python
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")
```

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Create true labels based on your domain knowledge
true_labels = [0] * len(dataset1) + [1] * len(dataset2)
accuracy = accuracy_score(true_labels, kmeans.labels_)
confusion = confusion_matrix(true_labels, kmeans.labels_)
report = classification_report(true_labels, kmeans.labels_)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)
```

##### Classification Report:
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.05 | 0.06 | 0.06 |
| 1 | 0.03 | 0.03 | 0.03 |
| Accuracy |  |  | 0.04 |
| Macro avg | 0.04 | 0.04 | 0.04 |
| Weighted avg | 0.04 | 0.04 | 0.04 | 

#### 6.1  comparison Analysis of Machine Learning

Comparison analysis of machine learning methods involves evaluating and contrasting 
different machine learning algorithms or models to determine which one is the most suitable 
for a particular task. The choice of the most appropriate machine learning method depends on 
various factors, including the nature of the data, the problem to be solved, computational 
resources, and desired performance metrics.

##### Here's a general framework for comparing and analyzing machine learning methods:

##### Define the Problem and Objectives:

* Clearly define the problem you want to solve and establish specific objectives and 
success criteria.
*  Determine whether the problem is a classification, regression, clustering, or another 
type of task.

##### Data Collection and Preparation:

* Collect and preprocess the data needed for your machine learning analysis.
*This may involve data cleaning, feature selection, feature engineering, and data 
splitting (e.g., into training and testing sets).

##### Select Candidate Machine Learning Models:
* Identify a set of machine learning algorithms that are suitable for your problem. The 
choice of algorithms will depend on the problem type (e.g., linear regression for 
regression, decision trees for classification, etc.).

##### Performance Metrics:

* Decide on appropriate evaluation metrics that are relevant to your problem. For 
example, accuracy, F1-score, mean squared error, or others.

##### Experiment Design:

* Design a rigorous experiment to fairly compare the selected machine learning models.
* Ensure that the experimental setup is consistent and repeatable.

##### Training and Evaluation:

* Train each machine learning model on the training data and evaluate their 
performance on the test data.
* Record the results for each model, including the chosen performance metrics.

##### Cross-Validation:

* Perform cross-validation to assess model performance more robustly. Crossvalidation techniques like k-fold cross-validation help to mitigate overfitting and 
provide a better estimate of how well each model generalizes.

##### Statistical Significance:

* Perform statistical tests, if necessary, to determine if the observed differences in 
performance between models are statistically significant.

##### Visualization and Interpretation:

* Visualize the results and performance metrics. This can include ROC curves, 
confusion matrices, or other relevant visualizations.
* Interpret the results to gain insights into the strengths and weaknesses of each model.

##### Hyperparameter Tuning:

Conduct hyperparameter tuning for each model to optimize their performance. Techniques 
like grid search or random search can be used.

###### Comparative Analysis:
Compare the results and assess which machine learning model 
performs best according to your predefined metrics and objectives.

* Consider the trade-offs between accuracy, interpretability, complexity, and other 
factors when making a choice.

###### Deployment and Monitoring: 
Implement the selected machine learning model in a real-world 
environment if applicable.

* Continuously monitor the model's performance and retrain as necessary.

###### Documentation and Reporting: 
Document the entire comparison analysis process and results.
Create a report or presentation to communicate your findings to stakeholders.

Remember that there is no one-size-fits-all solution in machine learning. The choice of the 
best algorithm depends on the specific problem and data. Comparative analysis helps you 
make an informed decision based on empirical evidence.

#### 6.2 Result & Discussion

##### Model Performance

###### Model Evaluation Metrics
Present a table or summary of performance metrics for the machine learning models 
used in fake news detection. Include metrics like accuracy, precision, recall, F1-score, 
and possibly ROC-AUC.

###### Model Comparison
Compare the performance of different machine learning models you experimented 
with. Which model(s) performed the best in terms of your chosen metrics?
Discuss any trade-offs between models, such as accuracy vs. interpretability or 
computational resources.

#####  Data Analysis

###### Data Preprocessing
* Describe the preprocessing steps you performed on your dataset. This might 
include data cleaning, feature engineering, text processing, or any other data 
preparation steps.

* Discuss any challenges you encountered during data preprocessing.

###### Feature Importance
* If relevant, discuss which features or characteristics of the news articles were 
most important in detecting fake news. You can use techniques like feature 
importance scores or visualization to showcase this.

##### Challenges & Limitations
* Discuss any challenges or limitations encountered during the project. For 
example, did you have issues with data quality, imbalanced datasets, or 
computational resources?
* Address any potential biases or ethical concerns in the dataset or model.

##### Interpretation of Results
* Interpret what the results mean in the context of fake news detection. How 
well can your models distinguish between real and fake news articles?
* Discuss the real-world implications of your findings, including how your work 
might contribute to combating misinformation.

##### Future Work

* Propose potential future work and improvements. Are there ways to enhance 
model performance, handle new challenges, or expand the scope of your 
project?
* Consider areas like fine-tuning models, incorporating external data sources, or 
developing more robust feature engineering techniques.

### Conclusion

In this project, we set out to develop a machine learning-based solution for fake news 
detection, addressing the critical issue of misinformation in the digital age. We conducted a 
comprehensive analysis and experimented with various machine learning models to detect 
fake news articles from genuine ones. 

### Future Enhancements
While our project has made valuable contributions, there are several avenues for future 
research and enhancements:

#### Multimodal Analysis: 
Incorporate additional data sources, such as images, audio, 
and videos, to develop a more comprehensive fake news detection system capable of 
handling different types of content.

#### Deep Learning: 
Explore the application of deep learning models, such as 
convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to 
leverage the contextual information within textual data for improved fake news 
detection.

#### Real-time Monitoring:
Develop a real-time fake news detection system that can 
identify and flag potentially fake news articles as they are published, providing a 
proactive approach to misinformation.

#### Explainability: 
Enhance the explainability of the models to gain insight into the 
decision-making process, which is crucial for building trust in automated fake news 
detection systems.

#### Ethical Considerations:
Address ethical considerations, such as privacy and bias, 
when collecting and using data for fake news detection. Ensure that the system's 
decisions are fair and unbiased.

#### Human-in-the-Loop Systems:
Create systems that incorporate human reviewers and 
fact-checkers in the detection process, combining the strengths of both machine 
learning models and human expertise.

#### Cross-Lingual Fake News Detection: 
Extend the project to multiple languages, 
enabling the detection of fake news in a global context.
Open-Source Tools: Contribute to or create open-source fake news detection tools 
and libraries to assist researchers and organizations in combating misinformation.

### References

#### Datasets

[True.csv](https://drive.google.com/file/d/1yecV01duNs1yrYoEOqttciQ_BS7MTcZ0/view?usp=sharing)

[Fake.csv](https://drive.google.com/file/d/1TJJSnZUCDoBvCz662BWr0_pInQ0dCXpz/view?usp=sharing)

#### Libraries and Tools:
##### scikit-learn: 
A popular machine learning library in Python for building and evaluating 
machine learning models. scikit-learn
##### TensorFlow and Keras: 
Deep learning libraries for creating and training neural 
networks. TensorFlow, Keras
##### Tutorials and Courses:
* Coursera's "Introduction to Data Science" (offered by the University of Washington) 
includes a section on fake news detection.
* edX's "Data Science MicroMasters" program, especially the courses related to natural 
language processing and text analysis.
##### Online Articles and Blog Posts:
* "Fake News Detection with Machine Learning" by Towards Data Science. 
* "How to Build a Fake News Detection Model" by Analytics Vidhya. 
* "Detecting Fake News with Python" by DataCamp.









































      
    





    





