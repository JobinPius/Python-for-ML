 # Feedback Analysis for Intel course

## Introduction

This feedback report presents an analysis of student satisfaction with regards to teacher's teaching quality, knowledge, and other relevant aspects. The feedback was collected through a survey administered to students, asking them to evaluate various aspects of their teachers' performance and their overall learning experience.

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

```python
#pip install seaborn
#df_class=pd.read_csv("/content/survey_data.csv")
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
df_class.head()
```

## Data wrangling

```python
df_class.info()
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
df_class.info()
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
df_class.sample(5)
# checking for null
df_class.isnull().sum().sum()
# dimension
df_class.shape
```


## Dataset overview
The dataset contains feedback responses from participants of a session. It includes several columns:

  * Timestamp: It includes the date and time when the feedback was submitted by the student.

  * Name of the Participant: The name of the participant who provided the feedback.

  * Email ID: The email address of the participant.

  * Branch: from which branch the student comes from.

  * Semester: The current semester of the participant.

  * Resource Person of the session: The resource person for the session.

  * Ratings: It is the rating provided by the participant for the teachers who conducted the session based on several factors like content quality,effectiveness 
             etc


## Exploratory Data Analysis (EDA)

   ```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Name"].value_counts(normalize=True)*100,2)
```

* Resource Person-wise Distribution: This report provides an analysis of the feedback collected from students regarding their satisfaction with resource persons 
   
* Participant Name Analysis:This report provides an analysis of participant names in the context of an educational program or event.

* Visualization: created to understand the faculty-wise distribution of data  across different categories
 
  ```python
  ```python
  ax = plt.subplot(1,2,1)
  ax = sns.countplot(x='Resourse Person', data=df_class)
  #ax.bar_label(ax.containers[0])
  plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
  ax =plt.subplot(1,2,2)
  ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
  ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
  ```
  ```python
  ax = plt.subplot(1,2,1)
  ax = sns.countplot(x='Resourse Person', data=df_class)
  #ax.bar_label(ax.containers[0])
  plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
  ax =plt.subplot(1,2,2)
  ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
  ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
  ```
![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/fc77d696-b6bd-41b9-b87a-b91929868c99)


## Content Quality:

 * This boxplot suggests that the majority of resource persons are consistently delivering high-quality content, as reflected by the median ratings.

 * the absence of outliers and the narrow interquartile range across all participants indicate a consistent trend of high ratings, underscoring the uniformity of 
   positive feedback.

  ```python
  sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
  plt.show()  
  ```
   ![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/5a9a5fc8-357c-43e8-bc0d-a35e41c4c8d9)
  
  
## Effectiveness:

 * The effectiveness of the sessions seems to be rated highly as well, with few outliers indicating occasional deviations from otherwise high ratings.
 * Mrs. Veena A Kumar and Mrs. Gayathri J L have outliers on the lower end, which might suggest specific sessions where the effectiveness was rated lower than 
   usual.

  ```python
  sns.boxplot(y=df_class['Resourse Person'],x=df_class['Effectiveness'])
  plt.show()
  ```
  ![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/735e316a-55d3-48fe-8aa3-87962ebbfe4b)

  
## Expertise:

 * Similar to effectiveness, expertise ratings are high with few outliers.

 * The outliers for Mrs. Veena A Kumar and Mrs. Gayathri J L suggest that there were certain sessions where the perceived expertise of the resource person 
   was not up to the participants' expectations.

   
  ```python
  sns.boxplot(y=df_class['Resourse Person'],x=df_class['Expertise'])
  plt.show()
  ```
  ![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/3d99ad6f-5bd9-488a-b56c-879f5d207c0b)

## Relevance: 

 * Relevance suggests that the  session content was well-aligned with the participants' needs and was perceived as highly relevant to real-world scenarios, as 
   reflected by the absence of outliers and consistently high median values.
 
 
  ```python
  sns.boxplot(y=df_class['Resourse Person'],x=df_class['Relevance'])
  plt.show()
  ```
  ![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/616c3662-9073-44dd-9dba-41413c76b07f)


## Ratings by Branch

This uniformity in tight interquartile ranges across all branches in content quality ratings, with only a few outliers, indicates a consistent perception of high content quality irrespective of the academic department.

 ```python
  sns.boxplot(y=df_class['Resourse Person'],x=df_class['Branch'])
  plt.show()
  ```

 ![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/57d59129-bb64-4ec7-9fdf-4c8616cae68e)

  


# Using K-means Clustering to identify segmentation over student's satisfaction

## For finding the best value of k using elbow method

```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
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

![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/41161fa2-0035-41e2-a73d-e3e77352b4a4)



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
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```

##### KMeans(n_clusters=3, n_init='auto', random_state=42)
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
![image](https://github.com/JobinPius/Python-for-ML/assets/160459885/f39e3b0b-ebd4-483c-aeda-edb1f7c15a28)

## Result and Conclusion

  * Understanding resource person performance through feedback can provide valuable insights into their strengths and areas for improvement, facilitating targeted 
    professional development efforts and ultimately enhancing the overall quality of future sessions.




