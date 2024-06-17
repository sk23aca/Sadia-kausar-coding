#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split

# Create sample data for clustering
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 1. Histogram
def plot_histogram(data):
    plt.hist(data, bins=30, alpha=0.7, color='blue')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# 2. Scatter Plot with Line Fitting
def plot_scatter_with_line(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted line')
    plt.title('Scatter Plot with Line Fitting')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# 3. Pie Chart
def plot_pie_chart(data):
    counts = data.value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('Pie Chart of Categories')
    plt.show()

# 4. Heatmap
def plot_heatmap(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Data Correlation')
    plt.show()

# 5. Bar Chart
def plot_bar_chart(data):
    counts = data.value_counts()
    counts.plot(kind='bar', color='skyblue')
    plt.title('Bar Chart of Categories')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.show()

# Example Data for Histogram, Heatmap, Pie Chart, and Bar Chart
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100)
})

# Example Categorical Data for Pie Chart and Bar Chart
category_data = pd.Series(np.random.choice(['Category 1', 'Category 2', 'Category 3'], size=100))

# Plotting
plot_histogram(data['A'])  # Histogram
plot_scatter_with_line(data['A'].values.reshape(-1, 1), data['B'].values)  # Scatter Plot with Line Fitting
plot_pie_chart(category_data)  # Pie Chart
plot_heatmap(data)  # Heatmap
plot_bar_chart(category_data)  # Bar Chart

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Silhouette Score
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Additional: Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Assuming y is the true labels and labels are the predicted clusters
plot_confusion_matrix(y, labels)


# In[ ]:




