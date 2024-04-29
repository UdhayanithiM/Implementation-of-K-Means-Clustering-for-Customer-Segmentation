# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Pick customer segment quantity (k).
2. Seed cluster centers with random data points.
3. Assign customers to closest centers.
4. Re-center clusters and repeat until stable.

## Program:

```python
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Udhayanithi M
RegisterNumber:  212222220054
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
# define colors for each cluster
colors = ['r', 'g', 'b', 'c', 'm']

# plotting the controls
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

  #Find minimum enclosing circle
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

#Plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure aspect ratio is equal
plt.show()
```

## Output
![image](https://github.com/Bhargava-Shankar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/85554376/cc740d26-2e3b-44d5-805f-ab5d65039f34)

![image](https://github.com/Bhargava-Shankar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/85554376/e42684f1-cf93-46a5-812f-b90db347ac6b)

![image](https://github.com/Bhargava-Shankar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/85554376/a94e35c8-bbe0-49bb-9248-756c42e05ed8)


![image](https://github.com/Bhargava-Shankar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/85554376/812ebdfd-b2f0-4dd7-93b1-462b432ae9f0)

![image](https://github.com/Bhargava-Shankar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/85554376/42cb5adf-286c-438c-9c72-97b0476d5de1)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
