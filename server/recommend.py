import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



songs  = pd.read_csv('songs_scaled.csv',index_col=0)


def retrieve_rows(song_list, songs_df):
    rows = []
    for x in song_list:   
        result = songs_df[songs_df['name'].str.contains(str(x), case=False, na=False)]
        if not result.empty:
            first_row = result.iloc[0]
            rows.append(first_row)
    new_df = pd.DataFrame(rows)
    return new_df



def get_user_interest_vector(User_History,Ratings) :
    weighted_row = np.zeros(1007)
    for x in range(len(User_History)) :
        row = (User_History.iloc[x,2:])
        row = np.array(row)
        weighted_row = weighted_row + row * Ratings[x]
    User_Profile = np.array(weighted_row / np.sum(weighted_row))
    return User_Profile




def get_pool_of_similar_songs(User_Interest,songs_df,pool_size) :
    Matrix  =  songs_df.iloc[:,2:]
    User_Interest = User_Interest.reshape(1,-1)
    Utility  = Matrix * User_Interest
    Sum_vector = np.sum(Utility, axis=1)
    Resultant = songs.copy(deep=True)
    Resultant['sum'] = Sum_vector
    Resultant = Resultant.sort_values(by='sum', ascending=False)
    return Resultant.iloc[:pool_size,]

def get_rating(song_name, ratings_df):
    # Searching for substring matches in 'name' column
    result = ratings_df[ratings_df['name'].str.contains(song_name, case=False)]
    
    if not result.empty:
        # Retrieve the first rating if a match is found
        rating = result.iloc[0]['Rating']
        return rating
    else:
        print("song_name = ",song_name)


def calculate_cosine_similarity(user_history_df, song_pool_df):    
    # Extracting values from DataFrames
    user_history_values = user_history_df.values
    song_pool_values = song_pool_df.values
    
    # Calculating cosine similarity matrix
    similarity_matrix = cosine_similarity(user_history_values, song_pool_values)
    
    return similarity_matrix


def recommender(User_ratings):
    Songs_l = User_ratings.iloc[:,0]
    Ratings = User_ratings.iloc[:,1]
    User_History = retrieve_rows(Songs_l,songs)
    print(User_History)
    User_Interest_Vector = get_user_interest_vector(User_History,Ratings)
    print(User_Interest_Vector)
    Songs_Pool = get_pool_of_similar_songs(User_Interest_Vector,songs,500)
    Clean_Pool = Songs_Pool.drop_duplicates(subset='name', keep='first')
    Clean_Pool.to_csv('Recommendations.csv')
    top_Header = Clean_Pool['name']
    Side_Header = User_History['name']
    Sim = calculate_cosine_similarity(User_History.iloc[:,2:], Clean_Pool.iloc[:,2:-1])
    cosine_similarity_df = pd.DataFrame(Sim, index=Side_Header, columns=top_Header)
    cosine_similarity_df.to_csv('Cosine_Similarity.csv')
    my_dict = {}  # Initialize an empty dictionary
    N = 2
    for col in range(cosine_similarity_df.shape[1]) :
        column_name = cosine_similarity_df.columns[col]
        Item_to_Item_cosine = cosine_similarity_df.iloc[:, col]  
        Item_to_Item_cosine.head()
        Item_to_Item_cosine = Item_to_Item_cosine.sort_values(ascending=False)
        Predicted_Rating = 0
        for x in range(N) :
            result_row = cosine_similarity_df[cosine_similarity_df.iloc[:, col] == Item_to_Item_cosine[x]]
            index = result_row.index[0]
            Predicted_Rating = Predicted_Rating + get_rating(str(index), User_ratings) * Item_to_Item_cosine[x]
            if x == N-1 : 
                my_dict[str(column_name)] = Predicted_Rating/N

#     # Printing each key-value pair in a loop
#     for key, value in my_dict.items():
#         print(key,"   ", value)
    top_10_values = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True)[:10])
#     for key, value in top_10_values.items():
#         print(f"Key: {key}, Value: {value}")
    return top_10_values






User_ratings = pd.read_csv('input.csv', index_col=0)
print(User_ratings)
top_10_values = recommender(User_ratings)
# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(top_10_values.items()), columns=['Key', 'Value'])
# Store the DataFrame as a CSV file
df.to_csv('output.csv', index=False)

