# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Create Mall Customer Dataset
data = {
    'CustomerID': list(range(1, 21)),
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male'],
    'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30,
            67, 35, 58, 24, 37, 22, 35, 20, 52, 35],
    'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
                           19, 19, 20, 20, 20, 20, 21, 21, 23, 23],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                               14, 99, 15, 77, 13, 79, 35, 66, 29, 98]
}

df = pd.DataFrame(data)

# Save original dataset
df.to_csv("C:\\Users\\Dell\\Downloads\\Mall Custemers data.csv", index=False)



# Step 2: Preprocess Data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find optimal k (Elbow Method) – Optional
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# Step 5: Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Save the model
joblib.dump(kmeans, "kmeans_model.pkl")

# Step 7: Save clustered dataset
df.to_csv("clustered_customers.csv", index=False)

# Step 8: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 2], centroids[:, 3], s=300, c='yellow', marker='X', label='Centroids')
plt.title('Customer Segments (Including Age & Gender)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_plot.png")
plt.show()

# Step 9: Print Cluster Averages
cluster_summary = df.groupby('Cluster')[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("Cluster Averages:\n", cluster_summary)
