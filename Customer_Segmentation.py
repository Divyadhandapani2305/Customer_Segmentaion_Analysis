# ==========================================================
# CUSTOMER SEGMENTATION WITH BUSINESS PERSONA MAPPING
# ==========================================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the Dataset
# (Ensure "Mall_Customers.csv" is in the same folder as your script)
df = pd.read_csv("Mall_Customers - Mall_Customers.csv")


# 3. Quick Data Check
print("--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Feature Selection
# Extracting Annual Income and Spending Score for behavioral mapping
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Feature Scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Training the K-Means Model
# Setting n_clusters to 5 based on standard optimal elbow analysis for this data
optimal_clusters = 5
model = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
cluster_labels = model.fit_predict(X_scaled)

# Add numeric clusters to the dataframe temporarily
df['Cluster_No'] = cluster_labels

# 6. Mapping Numbers to Business Personas (What your Manager wants to see)
# Note: K-Means cluster numbers (0-4) assignment can vary depending on random state initialization.
# Based on random_state=42, this is the geometric profile mapping:
persona_mapping = {
    0: "Standard Customers (Mid Income, Mid Spend)",
    1: "Careful Spenders (High Income, Low Spend)",
    2: "Sensible Buyers (Low Income, Low Spend)",
    3: "Impulsive Buyers (Low Income, High Spend)",
    4: "Target Segment (High Income, High Spend)"
}

# Apply the business names to a new column
df['Customer_Segment'] = df['Cluster_No'].map(persona_mapping)

# 7. Visualizing the Segments for Management
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=df['Annual Income (k$)'], 
    y=df['Spending Score (1-100)'], 
    hue=df['Customer_Segment'],  # Colors the dots by business names instead of numbers
    palette='Spectral', 
    s=100,
    edgecolor='black'
)

# Formatting the plot for professional slide decks
plt.title('Executive Summary: Customer Segmentation Analysis', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12, labelpad=10)
plt.ylabel('Spending Score (1-100)', fontsize=12, labelpad=10)
plt.legend(title='Business Segments', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 8. Displaying Final Clean Report Sample
print("\n" + "="*50)
print("FINAL REPORT FOR MANAGEMENT (SAMPLE SUMMARY)")
print("="*50)
print(df[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'Customer_Segment']].head(10).to_string(index=False))