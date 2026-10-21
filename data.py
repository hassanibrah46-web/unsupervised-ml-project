from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#prepare customers data using a python list
data = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14], 
    [13000, 11],
    [3000, 4]
]
#scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

#train the model
kmeans.fit(scaled_data)

#display the cluster for each customer
clusters = kmeans.labels_
print(f"customer cluster results:")

for i, cluster in enumerate(clusters):
    print(f"Customer {i+1}: Cluster {cluster}")

#display total customers in each cluster
print("\nTotal customers in each cluster:")

for cluster in range(3):
    total = list(clusters).count(cluster)
    print(f"Cluster {cluster}: {total} customers")
    

