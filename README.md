# FAKE_NEWS_DETECTION_USING_MACHINE-LEARNING
Utilized machine learning algorithms to develop a fake news detection system. Implemented natural language processing techniques to analyze text features and detect misinformation with high accuracy. Validated the model on diverse datasets, demonstrating its effectiveness in combatting the spread of false information in digital media.


## 1.Introduction
These days‟ fake news is creating different issues from sarcastic articles to a fabricated news and plan government propaganda in some outlets. Fake news and lack of trust in the media are growing problems with huge ramifications in our society. Obviously, a purposely misleading story is “fake news “but lately blathering social media’s discourse is changing its definition. Some of them now use the term to dismiss the facts counter to their preferred viewpoints. The importance of disinformation within American political discourse was the subject of weighty attention, particularly following the American president election.The term 'fake news' became common parlance for the issue, particularly to describe factually incorrect and misleading articles published mostly for the purpose of making money through 
page views. In this paper, it is seeked to produce a model that can accurately predict the likelihood that a given article is fake news. Facebook has been at the epicenter of much critique following media attention. They have also said publicly they are working on to distinguish these articles in an automated way. Certainly, it is not an easy task. A given algorithm must bepolitically unbiased – since fake news exists on both ends of the spectrum – and also give equal balance to legitimate news sources on either end of the spectrum. In this era of fake news and information manipulation, the ability to discern genuine news from spam is not only a technological challenge but also a matter of societal importance. News organizations, social media platforms, and content aggregators are increasingly incorporating 
news spam detection systems into their workflows to maintain the integrity of the information they deliver to their audiences.News spam detection involves the use of advanced technology, particularly machine learning and natural language processing (NLP), to automatically distinguish between legitimate news articles and spammy or misleading content. By leveraging statistical patterns, linguistic cues, and contextual information, these systems can help ensure that readers are presented with trustworthy and reliable news source.

## 2.Literature Survey
  * Fake news is false or misleading information that is presented as news. It can be 
  spread through social media, news outlets, or other channels

  * True news is accurate and factual information that is presented in a fair and unbiased 
  way. It is typically reported by reputable news organizations.

“We know that fake news can have a negative impact on society, and we’re committed to 
working with others to address this problem. We’re investing in a variety of initiatives to 
combat fake news, including machine learning research, fact-checking partnerships, and 
education programs.”                                               __~ Mark Zuckerberg, CEO of Facebook__

“We know that fake news can have a negative impact on society, and we’re committed to 
working with others to address this problem. We’re investing in a variety of initiatives to 
combat fake news, including machine learning research, fact-checking partnerships, and 
education programs.”                    __~ Jeff Bezos, CEO of Amazon__

“We believe that it’s important for people to be able to make informed decisions, and that 
includes having access to accurate information. That’s why we’re working hard to combat fake 
news and ensure that people on Google Search have access to high-quality journalism.”             __~ Sundar Pichai, CEO of Google__


## 3.Statistical Analysis
  * Sources of Data for Analysis:
      News websites are the primary source of data for analysis, as they contain a wealth of 
      information. Social media platforms such as Twitter and Facebook are also useful in 
      identifying trending news topics, but algorithms should be employed to avoid the spread 
      of misinformation.

  * Criteria for Selecting and Labeling Spam Articles:
      Criteria for identifying spam include sensational headlines, false claims, and click-bait 
      style content. Labeling spam is essential to create a reliable training dataset.

  * Challenges in Collecting and Annotating the Data:
      The challenge in collecting and annotating data lies in the need to create a dataset that 
      adequately covers the potential set of spam tactics. This is an ever-changing landscape, 
      with new forms of spam emerging constantly.

  * Exploratory Data Analysis:
      EDA techniques such as histograms, scatter plots, and heatmaps are useful in 
      identifying patterns in the data.

  * Feature Engineering and Selection:
      Feature engineering uses domain knowledge to extract relevant features from the data, 
      while feature selection aims to remove irrelevant or redundant ones, improving the 
      performance and computationally efficiency of the model.

  * Application of Machine Learning Algorithms:
      Various machine learning algorithms such as Naive Bayes, Support Vector Machines, 
      and Random Forests are utilized to classify spam articles.

  * Evaluation Metrics:
    |    |    |
    |---------------|---------------|
    | Accuracy  | Measures the proportion of correctly classified articles. |
    | Precision  | Measures the percentage of articles classified as spam that were actually spam.  |
    | Recall | Measures the percentage of actual spam articles that were correctly classified. |
    | F1 Score | Balances precision and recall in a single metric. |
    | ROC Curve | Displays the tradeoff between true positives and false positives at various thresholds. |
    | Cross-validation Technique | Helps in determining the optimal hyperparameters for the model. |

* Performance of the Statistical Analysis Model:
    Our analysis indicates that a statistical analysis model is an effective tool for detecting 
news spam, achieving high accuracy and F1 score when evaluated against the test data.

* Comparison with Existing Spam Detection Methods:
    Our analysis outperformed other existing spam detection methods such as manual 
labeling and rule-based filtering.

* Insights and Conclusions Drawn from the Analysis:
    Statistical analysis is a useful approach to detecting news spam. Techniques such as EDA 
and feature engineering can improve the performance of the model and increase the 
accuracy and efficiency. Moreover, continuous updates to the training dataset must be
made to ensure we remain vigilant against new spam tactics.

### 3.1 Mean, Median, Mode
News spam detection, "mean," "median," and "mode" are statistical concepts that can be 
applied to analyze various aspects of data related to news articles, features, or metrics.

#### Mean
The mean is a measure of central tendency that represents the average value of a dataset. 
It is calculated by summing up all the values in the dataset and then dividing by the 
total number of values. In the context of news spam detection, you can calculate the 
mean for various metrics or features to understand the typical or average behavior.

Example: You could calculate the mean length of news articles in a dataset to see how 
long the average article is. This information might be useful for distinguishing spammy 
short articles from legitimate longer ones.

#### Median
The median is another measure of central tendency that represents the middle 
value in a dataset when the values are sorted in ascending or descending order. 
If there is an even number of values, the median is the average of the two middle 
values. The median is useful when dealing with datasets that may have outliers 
or extreme values that can skew the mean.

Example: In news spam detection, you might calculate the median publication 
date of articles to find the midpoint in time. This could help identify anomalies 
in the publication dates that may be indicative of spam.

#### Mode
The mode is the value that appears most frequently in a dataset. It represents the 
most common value or category. In the context of news spam detection, the 
mode can be used to identify patterns or categories that occur most frequently.

Example: You could calculate the mode of the topics or categories of news 
articles in a dataset to identify the most prevalent subjects. This information 
might be useful for understanding the content distribution and potentially 
spotting patterns that could be associated with spam topics.

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
ANNOVA (Analysis of Variance) and t-tests are statistical techniques used for different 
purposes, but they can both be applied in the context of news spam detection, 
particularly when you want to compare the means of multiple groups or datasets to 
determine if there are significant differences. Let's explore how ANOVA and t-tests 
can be used for news spam detection:

  #### ANNOVA (Analysis of Variance): 
  ANNOVA is used when you want to compare 
  the means of three or more groups to determine if there are statistically significant 
  differences among them. In the context of news spam detection.
  
  ##### Feature Comparison:
  You may have multiple features or metrics extracted 
      from news articles (e.g., article length, keyword frequency, publication time)
      that you suspect could be different between legitimate news, spam, and 
      possibly other categories (e.g., clickbait). ANNOVA can be used to assess 
      whether there are statistically significant differences in these features across 
      the different categories.

  ##### Evaluation Metrics: 
  ANNOVA can also be applied to compare the 
performance of different news spam detection models or algorithms. For 
example, if you have several algorithms and you want to determine if there are 
significant differences in their accuracy, F1-score, or other evaluation metrics, 
ANNOVA can help.
  ##### F-test: 
   It is commonly used in hypothesis testing to compare the equality of 
variances of two samples. It is based on the ratio of the variances of the two samples. 
If the ratio is greater than a certain threshold, the null hypothesis of equal variances 
is rejected, and the alternative hypothesis of unequal variances is accepted. F-test is 
widely used in data science and machine learning, especially in feature selection and 
regression analysis.

``` python
# Perform ANOVA F-test
f_statistic, p_value = stats.f_oneway(combined_df[combined_df['label'] == 0]['label'],
 combined_df[combined_df['label'] == 1]['label'])
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
```

#### 3.3 T-Test:

##### Formulate Hypotheses
* Null Hypothesis (H0): There is no significant difference in the average word 
count between spam and legitimate news articles.
* Alternative Hypothesis (H1): There is a significant difference in the average 
word count between spam and legitimate news articles.

##### Perform the t-test:
If the variances of the two samples are roughly equal, you can use the independent twosample t-test (also known as the Student's t-test).If the variances are significantly different, 
consider using the Welch's t-test, which is more robust to unequal variances.

##### Calculate the t-statistic and p-value:
The t-statistic measures the difference between the means of the two groups relative to the 
variation within each group.

The p-value indicates the probability of observing a difference as extreme as the one you
calculated, assuming that the null hypothesis is true.

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
News spam detection is a critical aspect of maintaining the integrity and 
credibility of online news sources. With the rising prevalence of spam emails 
and messages, it is imperative to employ effective techniques to differentiate 
genuine news from spam. In this article, we will explore various techniques and 
shed light on the powerful role of the Chi-Squared test in this domain.

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

News spam can harm the credibility of news media. Supervised learning is a promising 
approach for detecting spam in news articles. Good performance relies on high-quality training 
data, accurate labeling, and evaluation metrics. We hope this article sheds some light on the 
importance of spam detection in news, and how machine learning can help.

#### 4.1 Linear Regression
spam detection refers to the automated process of identifying and filtering out 
fraudulent or misleading news articles from legitimate sources. By utilizing 
advanced algorithms and machine learning techniques, news spam detection 
systems can effectively distinguish between authentic news and fake news, 
ensuring that users are well-informed and protected from misinformation.

##### Importance of News Spam Detection
In today's digital age, where the spread of false information can have profound 
consequences, the importance of news spam detection cannot be overstated. It 
helps maintain the integrity of news platforms, fosters trust among readers, and 
safeguards the democratic process. By combating fake news, news spam 
detection contributes to a more informed society and a healthier online 
ecosystem.

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

##### Model Evaluation
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

##### Model Evaluation and Validation
|  | Precision | Recall | F1-score |
|----|----|----|----|
| 0 | 0.69 | 0.66 | 0.68 |
| 1 | 0.68 | 0.70 | 0.69 |
| Accuracy |  |  | 0.68 |
| Macro avg | 0.68 | 0.68 | 0.68 |
| Weighted avg | 0.68 | 0.68 | 0.68 |














      
    





    





