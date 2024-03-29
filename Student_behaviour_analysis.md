 # Student behaviour Analysis for the Intel course

## Introduction 

The student behavior report provides an in-depth analysis of absenteeism, along with raised hands and other pertinent aspects. The dataset includes various features related to student behavior, encompassing demographic information such as gender, nationality, and place of birth, as well as academic details like stage, grade, section, topic, and semester. Additionally, it incorporates engagement metrics such as raised hands, visited resources, announcements viewed, and participation in discussions, along with parental engagement factors like parent survey responses and school satisfaction. Moreover, the dataset contains absence records indicating student absence days, contributing to a comprehensive understanding of absenteeism within the context of student behavior.

Once we have summarized the dataset's statistical properties, checked for missing values, and analyzed the distributions of numerical and categorical features, we can proceed with exploratory data analysis (EDA) and k-means clustering. This approach aids in identifying patterns, outliers, and the underlying structure of the data, thereby informing our strategy for implementing k-means clustering effectively.


## Importing necessary libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

## Loading Data
### Data Summary

The dataset consists of 480 entries with no missing values across all features. A brief summary of the numerical features includes:

- *Raised Hands*: Average ~ 46.78, Min 0, Max 100, Std Dev 30.78
- *Visited Resources*: Average ~ 54.80, Min 0, Max 99
- *Announcements View*: Average ~ 37.92, Min 0, Max 98
- *Discussion*: Average ~ 43.28, Min 1, Max 99


```python
#pip install seaborn
df_class=pd.read_csv("KMEANS_SAMPLE.csv")
df_class.head()
```

## Data wrangling
### Categorical Data Distributions

- *Gender*: Balanced distribution between male (M) and female (F) students.
- *Nationality and Place of Birth*: Wide range including Kuwait, Jordan, Palestine, Iraq, Lebanon, Egypt, among others.
- *StageID*: Includes lower level, middle school, and high school.
- *GradeID*: Ranges from G-02 to G-12.
- *SectionID*: Distributed across sections A, B, and C.
- *Topic*: Subjects include IT, Math, Arabic, Science, English, etc.
- *Semester*: Covers both the first (F) and second (S) semesters.
- *Relation*: Indicates the main respondent for the survey (mother/father).
- *ParentAnsweringSurvey*: Yes/No responses.
- *ParentschoolSatisfaction*: Satisfaction levels (Good/Bad).
- *StudentAbsenceDays*: Categorized as "Under-7" or "Above-7" days.
- *Class*: Performance categories labeled as Low (L), Middle (M), and High (H).


```python
df_class.info()
level_mapping = {'H': 2, 'M': 1, 'L': 0}
df_class['Class_num'] = df_class['Class'].map(level_mapping)
print(df_class)
df_class.info()
df_class.sample(5)
# checking for null
df_class.isnull().sum().sum()
# dimension
df_class.shape
```




## Exploratory Data Analysis (EDA)
#### Numerical Features Distribution Insights:

- *Raised Hands*: The distribution shows a bimodal pattern, suggesting two distinct groups of students based on classroom participation.
- *Visited Resources*: Distribution is somewhat bimodal, indicating varying levels of engagement with resources.
- *Announcements View*: Skewed towards lower frequencies, suggesting that many students do not regularly check announcements.
- *Discussion*: More uniform across the range but shows a slight skew towards lower engagement levels.


 ```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Class"].value_counts(normalize=True)*100,2)
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Topic"].value_counts(normalize=True)*100,2)
```

* Resource Person-wise Distribution: This report provides an analysis of the feedback collected for showing behaviour of students
   
* Participant Name Analysis:This report provides an analysis of participant names in the context of an educational program or event.

* Visualization: created to understand the faculty-wise distribution of data  across different categories
 
  
  ```python
  ax = plt.plot(1,2,1)
  ax = sns.countplot(x='Class_num', data=df_class)
  #ax.bar_label(ax.containers[0])
  plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
  ax.set_title(label = "Class", fontsize = 20,color='Brown',pad=20);
  ```![download](https://github.com/Christojs/Python-for-ML/assets/133338993/9eee6ca3-d807-4715-8369-91bc227a6090)

![download](https://github.com/Christojs/Python-for-ML/assets/133338993/9eee6ca3-d807-4715-8369-91bc227a6090)




## Gender:



  ```python
  sns.boxplot(y=df_class['Class'],x=df_class['gender'])
  plt.show()  
  ```
   ![download](https://github.com/Christojs/Python-for-ML/assets/133338993/8dda9a52-e33f-45ad-a7f5-00d3e6d231e8)

  
  
## Topic:



  ```python
  sns.boxplot(y=df_class['Class'],x=df_class['Topic'])
  plt.show()
  ```
  ![download](https://github.com/Christojs/Python-for-ML/assets/133338993/765add69-fb5c-4759-991f-dca8913733d1)


  
## Raisedhands:


   
  ```python
  sns.boxplot(y=df_class['Class'],x=df_class['raisedhands'])
  plt.show()
  ```
  ![download](https://github.com/Christojs/Python-for-ML/assets/133338993/87368087-108e-490b-a77f-85b7ce977b6a)


## Student absence days: 

 * Relevance suggests that the  session content was well-aligned with the participants' needs and was perceived as highly relevant to real-world scenarios, as 
   reflected by the absence of outliers and consistently high median values.
 
 
  ```python
  sns.boxplot(y=df_class['Class'],x=df_class['StudentAbsenceDays'])
  plt.show()
  ```
  ![download](https://github.com/Christojs/Python-for-ML/assets/133338993/0c9cc319-ecfe-4b95-b16d-b72339602bf4)



## Discussion



 ```python
  sns.boxplot(y=df_class['Class'],x=df_class['Discussion'])
  plt.show()
  ```
 ![download](https://github.com/Christojs/Python-for-ML/assets/133338993/eb27986d-a603-4c1a-abe6-f826f90b2f3e)


  


# Using K-means Clustering to identify segmentation over student's behaviour

#### Preparing for K-Means Clustering

Given the varied scales of the numerical features, standardizing the data is crucial before clustering to ensure each feature contributes equally to the distance calculations. We will explore the optimal number of clusters using the elbow method and perform k-means clustering on the standardized data.


## For finding the best value of k using elbow method

```python
input_col=["raisedhands","AnnouncementsView","Discussion","Class_num",]
X=df_class[input_col].values
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()

```

![download](https://github.com/Christojs/Python-for-ML/assets/133338993/b32bd92c-4c4e-4909-8445-29b47aed94ec)




## Using Gridsearch method

```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```

## Implementing K-means clustering

```python
# Perform k-means clustering
k = 2 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```

##### KMeans(n_clusters=2, n_init='auto', random_state=42)
##### In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
##### On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.


## Extracting labels and cluster centers

```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
df_class.head()
```

## Visualizing the clustering using first two features

```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![download](https://github.com/Christojs/Python-for-ML/assets/133338993/4cb5018d-6777-4009-a141-0ca3ca413ec8)

## Result and Conclusion
The variability observed in student engagement metrics during the exploratory analysis underscores the potential for clustering to uncover distinct patterns or groups of similar student behaviors. Consequently, the next phase entails implementing k-means clustering to segment the student population according to their engagement levels. Subsequently, these clusters will be analyzed in depth to extract actionable insights that can inform targeted interventions and improvements in educational outcomes.



