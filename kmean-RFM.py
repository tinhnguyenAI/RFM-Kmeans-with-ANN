import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix,classification_report, roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from itertools import cycle

# Importing the dataset
df = pd.read_csv('bigml_59c28831336c6604c800002a.csv')

# Encoding categorical data

labelencoder = LabelEncoder()
df['state'] = labelencoder.fit_transform(df['state'])
df['international plan'] = labelencoder.fit_transform(df['international plan'])
df['voice mail plan'] = labelencoder.fit_transform(df['voice mail plan'])
df['churn'] = labelencoder.fit_transform(df['churn'])

#show quantifier object churn and nonchurn
df["churn"].value_counts().plot.bar(title= 'Churn rate')
#Correlation heatmap
import seaborn as sns
f,ax= plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,linewidths=.6,fmt='.2f',ax=ax)
plt.show()

df['total minutes'] = df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes']
df['total calls'] = df['total day calls'] + df['total eve calls'] + df['total night calls'] + df['total intl calls']
df['avgcall'] = df['total minutes']/df['total calls']
df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']

#confirm RFM
rfmTable = df.groupby('phone number').agg({'total minutes':lambda x: x.sum(),
                                          #'total calls': lambda x: x.sum(), 
                                          'total charge': lambda x: x.sum()})
rfmTable.rename(columns={'total minutes': 'frequency',
                        # 'total calls': 'frequency', 
                         'total charge': 'monetary'}, inplace=True)

print(rfmTable.head())

# Create f_score

f_score = []
m_score = []
#r_score = []

columns = ['frequency', 'monetary']
scores_str = ['f_score', 'm_score']
scores = [f_score, m_score]

for n in range(len(columns)):
    # Order by column
    rfmTable = rfmTable.sort_values(columns[n], ascending=False)
    
    # Create new index
    refs = np.arange(1,3334)
    rfmTable['refs'] = refs
    
    # Add score
    for i, row in rfmTable.iterrows():
        if row['refs'] <= 666:
            scores[n].append(5)
        elif row['refs'] > 666 and row['refs'] <= 666*2:
            scores[n].append(4)
        elif row['refs'] > 666*2 and row['refs'] <= 666*3:
            scores[n].append(3)
        elif row['refs'] > 666*3 and row['refs'] <= 666*4:
            scores[n].append(2)
        else: 
            scores[n].append(1)

    # Create f_score column
    rfmTable[scores_str[n]] = scores[n]

rfmTableScores = rfmTable.drop(['frequency', 'monetary', 'refs'], axis=1)

print(rfmTableScores.head(5))

#from sklearn import preprocessing
    
#X = preprocessing.MinMaxScaler().fit_transform(X)
#X
#we will normalize our data so the prediction on all features will be at the same scale
X = rfmTableScores.iloc[:,[0,1]].values
    
X_train = preprocessing.MinMaxScaler().fit_transform(X)
#nurmalize the data
#from sklearn.preprocessing import StandardScaler
#X_std = StandardScaler().fit_transform(X)
#dfNorm = pd.DataFrame(X_std, index=rfmTableScores.index, columns=rfmTableScores.columns[0:2])

#X_train = dfNorm.iloc[:,0:2].values

#Kmeans k = 1,2,...,11
#build model k-means

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    
#elbow find k clusters optimal
plt.plot(range(1,11), wcss)
plt.title('Elbow graph')
plt.xlabel('Cluster number')
plt.ylabel('Total within-Cluster sum of squares')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(X_train)
kmeans_3 = kmeans.inertia_
centroids = kmeans.cluster_centers_
print(centroids)

kmeans_1 = KMeans(n_clusters=1, init='k-means++', random_state=0)
clusters_1 = kmeans_1.fit_predict(X_train)
kmeans_1 = kmeans_1.inertia_

espsilon = kmeans_3/kmeans_1
Accuracy = 1 - espsilon
print("Accuracy Kmeans: %.2f" %Accuracy)

#show clusters of object
rfmTable['clusters'] = clusters
print(rfmTable.head())

#statistical describe
print('statistical describe of cluster 0:')
print(rfmTable[(rfmTable['clusters'] ==0) ].describe())
print('statistical describe of cluster 1:')
print(rfmTable[(rfmTable['clusters'] ==1) ].describe())
print('statistical describe of cluster 2:')
print(rfmTable[(rfmTable['clusters'] ==2) ].describe())


#count cluster value
rfmTable["clusters"].value_counts().plot.bar(title= 'Clusters')
import collections
print(collections.Counter(clusters))

#plot FM in 2D


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

ax.scatter(rfmTable.frequency, rfmTable.monetary, s=10)

ax.set_xlabel('Frequency')
ax.set_ylabel('Monetary')

#plot clusters FM in 2D
fig = plt.figure(figsize=(10,8))
dx = fig.add_subplot(111)
#colors = ['blue', 'yellow', 'green', 'red']
colors = ['blue', 'green','orange']

for i in range(0,3):
    dx.scatter(rfmTable[rfmTable.clusters == i].frequency, 
               rfmTable[rfmTable.clusters == i].monetary, 
               c = colors[i], 
               label = 'Cluster ' + str(i), 
               s=10)
dx.set_title('Clusters of clients')
dx.set_xlabel('Frequency')
dx.set_ylabel('Monetary')
dx.legend()
