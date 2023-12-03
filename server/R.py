import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials



# Loading data
data = pd.read_csv("Songs_scaled.csv")
input_data = pd.read_csv('input.csv')




# Clustering
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(n_clusters=50))
])



X = data.iloc[:, 3:]  
song_cluster_pipeline.fit(X)


def get_sum_vector(song_list, spotify_data):
    song_sum_vector = np.zeros((1,spotify_data.shape[1] - 3))
    print(spotify_data.shape[1]-3)
    for song_name in song_list:
        song_data_row = spotify_data[spotify_data['name'] == song_name['name']].head(1)
        if not song_data_row.empty:
            song_vector = np.array(song_data_row.iloc[:, 3:].values)
            print(song_vector.shape[1])# Extract song vector for the song name
            song_sum_vector += song_vector
    return song_sum_vector



# Perform KMeans clustering on columns 3 to 12 to create 40 clusters
X_cluster = data.iloc[:, 3:]  # Columns 3 to 12 for clustering
song_cluster_pipeline.fit(X_cluster)


# Function to recommend songs
def recommend_songs(song_list, spotify_data, n_songs=30):
    metadata_cols = ['name']
    
    # Get summed vector of input songs
    summed_vector = get_sum_vector(song_list, spotify_data)

    # Scaling the summed song vector
    scaler = StandardScaler()
    scaler.fit(X)  # Fit the scaler on your data (X)
    scaled_sum_vector = scaler.transform(summed_vector.reshape(1, -1))

    # Compute cosine distances from summed vector to all cluster centers
    distances = cdist(scaled_sum_vector, song_cluster_pipeline.named_steps['scaler'].transform(song_cluster_pipeline.named_steps['kmeans'].cluster_centers_), 'cosine')

    closest_cluster_idx = np.argmin(distances)

    closest_cluster_songs = data[song_cluster_pipeline.named_steps['kmeans'].labels_ == closest_cluster_idx]
    
    rec_songs = closest_cluster_songs.head(n_songs)

    rec_songs = rec_songs[~rec_songs['name'].isin([song['name'] for song in song_list])]
    return rec_songs[metadata_cols]


# Generating recommendations
recommended_songs = recommend_songs(input_data.to_dict(orient='records'), data)
recommended_songs.to_csv('output.csv', index=False)
print("Done")