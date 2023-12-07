import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Loading data
data = pd.read_csv("Songs_scaled.csv")
input_data = pd.read_csv('input.csv')



# KMeans clustering without using Pipeline
scaler = StandardScaler()
X = data.iloc[:, 3:]
X_scaled = scaler.fit_transform(X)



# Elbow Method to determine optimal number of clusters
distortions = []



optimal_k =370
# KMeans clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)




# Silhouette score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")



# Davies-Bouldin index
db_index = davies_bouldin_score(X_scaled, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")



# Function to get summed vector of input songs
def get_sum_vector(song_list, spotify_data):
    song_sum_vector = np.zeros((1, spotify_data.shape[1] - 3))
    for song_name in song_list:
        song_data_row = spotify_data[spotify_data['name'] == song_name['name']].head(1)
        if not song_data_row.empty:
            song_vector = np.array(song_data_row.iloc[:, 3:].values)
            song_sum_vector += song_vector
    return song_sum_vector



def recommend_songs(song_list, spotify_data, n_songs=30, n_clusters=5, songs_per_cluster=6):
    metadata_cols = ['name']
    
    # Get summed vector of input songs
    summed_vector = get_sum_vector(song_list, spotify_data)

    # Scaling the summed song vector
    scaled_sum_vector = scaler.transform(summed_vector)

    # Compute cosine distances from scaled summed vector to all cluster centers
    distances = cosine_similarity(scaled_sum_vector, scaler.transform(kmeans.cluster_centers_))
    closest_cluster_indices = np.argsort(distances.flatten())[::-1][:n_clusters]

    recommended_songs = pd.DataFrame(columns=spotify_data.columns)
    for cluster_idx in closest_cluster_indices:
        cluster_songs = spotify_data[kmeans.labels_ == cluster_idx]
        
        # Calculate cosine similarity between the songs in the cluster and the summed vector
        cluster_song_vectors = scaler.transform(cluster_songs.iloc[:, 3:].values)
        cluster_distances = cosine_similarity(scaled_sum_vector, cluster_song_vectors)
        
        # Get the indices of the top 6 closest songs within the cluster
        closest_song_indices = np.argsort(cluster_distances.flatten())[::-1][:songs_per_cluster]
        
        # Add these top songs to the recommended list
        recommended_songs = pd.concat([recommended_songs, cluster_songs.iloc[closest_song_indices]])

    # Filter out input songs from the recommendations
    rec_songs = recommended_songs[~recommended_songs['name'].isin([song['name'] for song in song_list])]
    
    # Remove duplicates
    rec_songs = rec_songs.drop_duplicates(subset='name')
    
    return rec_songs[metadata_cols].head(n_songs)



# Generating recommendations
recommended_songs = recommend_songs(input_data.to_dict(orient='records'), data)

recommended_songs.to_csv('output.csv', index=False)
print("Done")
