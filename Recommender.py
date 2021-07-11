import numpy as np
import pandas as pd

class popularity_recommender():
    def __init__(self):
        self.train_data = None
        self.popularity_recommendations = None
        self.user_id = None
        self.song_id = None
        
    def create_list(self, train_data, user_id, song_id):
        self.user_id = user_id
        self.song_id = song_id
        self.train_data = train_data

        train_data_grouped = train_data.groupby([self.song_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)

        train_data_sort = train_data_grouped.sort_values(['score', self.song_id], ascending = [0,1])

        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id

        user_recommendations = user_recommendations[user_recommendations.columns.tolist()[-1:] + user_recommendations.columns.tolist()[:-1]]
        
        return user_recommendations
    

class item_similarity_recommender():
    def __init__(self):
        self.train_data = None
        self.cooc_matrix = None
        self.songs_dict = None
        self.song_id = None
        self.rev_songs_dict = None
        self.user_id = None
        self.item_similarity_recommendations = None

    def get_item_users(self, item):
        key = self.train_data[self.song_id] == item
        item_data = self.train_data[key]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users

    def get_user_items(self, user):
        key = self.train_data[self.user_id] == user
        user_data = self.train_data[key]
        user_items = list(user_data[self.song_id].unique())
        
        return user_items

    
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.song_id].unique())
            
        return all_items
    
    def construct_cooc_matrix(self, user_songs, all_songs):
        songs_users = []        
        for i in range(0, len(user_songs)):
            songs_users.append(self.get_item_users(user_songs[i]))
        #creating an empty matrix of the required size    
        cooc_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        for i in range(0,len(all_songs)):
            key = self.train_data[self.song_id] == all_songs[i]
            songs_i_data = self.train_data[key]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                users_j = songs_users[j]
                users_intersection = users_i.intersection(users_j)

                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    len1 = len(users_intersection)
                    len2 = len(users_union)
                    cooc_matrix[j,i] = float(len1)/float(len2)
                else:
                    cooc_matrix[j,i] = 0

        return cooc_matrix


    def generate_top_recommendations(self, user, cooc_matrix, all_songs, user_songs):
        print("Non zero values :%d" % np.count_nonzero(cooc_matrix))

        user_sim_scores = cooc_matrix.sum(axis=0)/float(cooc_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        df = pd.DataFrame(columns=['user_id', 'song', 'score', 'rank'])
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank += 1
        
        if df.shape[0] == 0:
            print("user has no songs for trainingitem similarity based recommendation model.")
            return -1
        else:
            return df

    def create_list(self, train_data, user_id, song_id):
        self.user_id = user_id
        self.song_id = song_id
        self.train_data = train_data

    def recommend(self, user):
        user_songs = self.get_user_items(user)       
        print("No. of unique songs: %d" % len(user_songs))
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooc_matrix = self.construct_cooc_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooc_matrix, all_songs, user_songs)
                
        return df_recommendations

    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooc_matrix = self.construct_cooc_matrix(user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooc_matrix, all_songs, user_songs)
         
        return df_recommendations


# Main Interractive Function

song_df_1 = pd.read_csv('triplets_file.csv')
song_df_2 = pd.read_csv('song_data.csv')
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
song_df['song'] = song_df['title']+' - '+song_df['artist_name']
song_df = song_df.head(100)

#Popularity Based Recommendation
#pr = popularity_recommender()
#pr.create_list(song_df, 'user_id', 'song')
#pr.recommend(song_df['user_id'][10])

#Similarity Based Recommendation
#ir = item_similarity_recommender()
#ir.create_list(song_df, 'user_id', 'song')
#ir.recommend(song_df['user_id'][5])
#print(ir.get_similar_items(['Oliver James - Fleet Foxes', 'The End - Pearl Jam']))

print("SONG RECOMMENDER--------------------\n Choose one : 1. Popularity Based    2. Similarity Based   3. Similar Songs\n Your Choice: ")
ans1 = int(input())

if(ans1 == 1):
    userno = int(input("Enter user no: "))
    pr = popularity_recommender()
    pr.create_list(song_df, 'user_id', 'song')
    print(pr.recommend(song_df['user_id'][userno]))

elif(ans1 == 2):
    userno = int(input("Enter user no: "))
    ir = item_similarity_recommender()
    ir.create_list(song_df, 'user_id', 'song')
    print(ir.recommend(song_df['user_id'][userno]))

elif(ans1 == 3):
    songs = []
    flag = 1
    print("Enter songs in the format \"title - artist\". enter -1 when you're done.\n")
    while(flag != 0):
        inp = input("Enter song: ")
        if(inp == "-1"):
            flag = 0
            break
        else:
            songs.append(input)
    print(ir.get_similar_items(songs))