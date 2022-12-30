import requests
import lyricsgenius
import json
import pandas as pd
import numpy as np 
import gensim
import string
import sklearn
from IPython.display import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from related_artist import get_related_artists
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from tqdm import tqdm
from gensim.test.utils import get_tmpfile
import time
import pickle
import os
from pandas import HDFStore
tqdm.pandas(desc="progress-bar")

token = "R5iNvu7MuLy8bUWhAY3DTbxpeAGGqj-83JfN7LO2TT98sZTli52wSHSIX1KjIDWl"

genius = lyricsgenius.Genius(token,excluded_terms=["Translations"], timeout=15, retries=3)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True 


## For an inputted artist, 
## save the lyrics of their top 100 songs in order to conduct analysis later. 
##Note that the inputted artists should come from the command line. 


## Idea is we store all of the lyrics as indices in a long list, 
##Later we abstract the indices as individual songs



## saved_data 


#saved_data = pd.DataFrame(columns=["Artist","song_name","Lyrics"])

saved_data = pd.read_hdf("./RYM-ML-main/Genuis-API/the_df.h5","s")
print("llol")
print(saved_data)

"""print("llol")
print(pd.read_pickle("./RYM-ML-main/Genuis-API/the_df.pkl"))
saved_data = pd.read_pickle("./RYM-ML-main/Genuis-API/the_df.pkl")"""
max_songs = 10

def from_song():
    start_time = time.time()
    song_input = input("Song: ")
    artist_input = input("Artist: ")
    song_object = genius.search_song(song_input,artist_input)
    print("--- %s seconds from_song ---" % (time.time() - start_time))
    return [song_object,artist_input]






def lyrics(given_song,artist):
    global saved_data
    print("ssui")
    start_time = time.time()
    lyrics_storage = []
    song_name = []
    if artist in saved_data['Artist'].values and given_song.title in saved_data["song_name"] :
        df = saved_data[saved_data["Artist"]]
        return [df,artist]
    
    
    artist_object = genius.search_artist(artist, max_songs=50,per_page=50, sort="popularity", include_features=False)

    for i in artist_object.songs:
        #Look into what type artist_object.song returns
        i.lyrics = i.lyrics.replace("\\"," ")
        lyrics_storage.append([i.lyrics])
        song_name.append(i.title)
    given_song.lyrics = given_song.lyrics.replace("\\"," ")
    lyrics_storage.append([given_song.lyrics])
    song_name.append(given_song.title)
    df = pd.DataFrame(lyrics_storage,columns=["Lyrics"])
    df["Artist"] = [artist for i in range(len(song_name))]
    df["song_name"] = song_name
    #saved_data = df

    print("--- %s seconds lyrics ---" % (time.time() - start_time))
    #saved_data.to_hdf("./RYM-ML-main/Genuis-API/the_df.h5","s")



    return [df,artist]



def cleaner(line):
    start_time = time.time()
    split = line.split(" ")
    new_list = []
    for i in split:
        i=i.lower()
        i=i.replace('"','')
        i = i.translate(str.maketrans('', '', string.punctuation))
        new_list.extend(i.split())

    print("--- %s seconds cleaner ---" % (time.time() - start_time))
    
    return " ".join(new_list)


def apply_cleaner(df):
    df["Lyrics"] = df["Lyrics"].apply(cleaner)
    return df


def tokenize_text(text):
    start_time = time.time()
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    
    print("--- %s seconds tokenize---" % (time.time() - start_time))
    return tokens

def get_tokens(df):
    return df.apply(lambda r: TaggedDocument(words=tokenize_text(r['Lyrics']), tags=[r.Artist,r.song_name]), axis=1)



def train_model(data):
    start_time = time.time()
    model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    
    print("--- %s seconds trainer--" % (time.time() - start_time))
    return model 





def vector_Collector(model,df):
    start_time = time.time()
    unique_songs  = []
    print("checker")
    print(df["song_name"])

    for songname in df["song_name"]:
        unique_songs.append(songname)

   



    song_hold = []
    for i in unique_songs: 
        try: 
            song_hold.append(model[i])
        except KeyError as e: 
            continue

    
    print("--- %s seconds vector_collector ---" % (time.time() - start_time))

   

    return [song_hold,unique_songs]






             

def prediction_matrix(song_vectors,model,unique_songs):
    start_time = time.time()
    kmeans = KMeans(n_clusters=4).fit(song_vectors)
    kmeans.predict(song_vectors)
    kmeans.cluster_centers_
    song_vectors = np.array(song_vectors)



    data_embed=TSNE(n_components=2, verbose=2, method='barnes_hut').fit_transform(song_vectors)
    df = pd.DataFrame(data_embed,columns=["x","y"])
    df["songs"] = unique_songs
    df["Cluster group"] = kmeans.labels_
    print(df)
    
    print("--- %s seconds predictor ---" % (time.time() - start_time))
    return df








    
def related_artist_matrix (artist_input,artist_prediction_matrix,initial_model):
    global saved_data
    print("sui")
    print(saved_data)
    lst = get_related_artists(artist_input)[:3]
    lyric_tot = []
    song_name = []
    artist_list=[]
    starter_dataframe =  artist_prediction_matrix.copy()
    starter_dataframe["Rapper"] = artist_input

    for artist_l in lst:
        print("messi")
        print(saved_data['Artist'].values)
        if artist_l in saved_data['Artist'].values:
            print({"is it ":True})
            the_table = saved_data[saved_data["Artist"]==artist_l]
            artist_list.extend([i for i in the_table["Artist"]])
            song_name.extend([i for i in the_table["song_name"]])
            lyric_tot.extend([i for i in the_table["Lyrics"]])
            continue

        elif artist_l not in saved_data['Artist'].values :
            print("no no")
            artist_object = genius.search_artist(artist_l, max_songs=50,per_page=50, sort="popularity",include_features=False)
            for i in artist_object.songs:
            #Look into what type artist_object.song returns
                i.lyrics = i.lyrics.replace("\\"," ")
                lyric_tot.append(i.lyrics)
                song_name.append(i.title)
                artist_list.append(artist_l)



    if artist_input in saved_data["Artist"].values :  
        print("ronaldo")
        the_saver = pd.read_hdf("./RYM-ML-main/Genuis-API/the_df.h5")
        the_table = the_saver[the_saver["Artist"]==artist_input]
        print(the_table)
        artist_list.extend([str(i) for i in the_table["Artist"]])
        song_name.extend([i for i in the_table["song_name"]])
        lyric_tot.extend([i for i in the_table["Lyrics"]])

    else: 
        last = genius.search_artist(artist_input, max_songs=50,per_page=50, sort="popularity",include_features=False)
        for w in last.songs:
        #Look into what type artist_object.song returns
            w.lyrics = w.lyrics.replace("\\"," ")
            lyric_tot.append(w.lyrics)
            song_name.append(w.title)
            artist_list.append(artist_input)


    df = pd.DataFrame()
    df["Lyrics"]= [str(i) for i in lyric_tot]
    df["Artist"] = artist_list
    df["song_name"] = song_name


    saved_data = saved_data.append(df)
    print("fif")
    print(saved_data)


    saved_data = saved_data.drop_duplicates()
    saved_data.to_hdf("./RYM-ML-main/Genuis-API/the_df.h5",key="s",format='t')

    print("vega")
    print(pd.read_hdf("./RYM-ML-main/Genuis-API/the_df.h5"))

    cleaned =apply_cleaner(df)
    print(cleaned)
    tok = get_tokens(cleaned)
    initial_model = train_model(tok)
    vec = vector_Collector(initial_model,df)

    per_artist_pred = prediction_matrix(vec[0],initial_model,vec[1])
    per_artist_pred["Rapper"] = artist_list
    print("fancy clown")

    print(per_artist_pred)

    return per_artist_pred






def viz (df):
 
    
    labels = df["Cluster group"]
    fig = px.scatter(df, x='x', y='y', color='Cluster group', size_max=6 ,text= "songs" , opacity=0.7)
    fig.update_traces(mode = 'markers')
    fig.show()

    return fig.show()
 



def visualization_multi (df):
    # model.labels_ is nothing but the predicted clusters i.e y_clusters

    ## General process. 
    ## Find that related artists, then union each data frame to the current one.
    ## Then you can use fig to visualize it. 
    
    labels = df["Rapper"]
    fig = px.scatter(df, x='x', y='y', color='Rapper', size_max=6 ,text= "songs" , opacity=0.7)
    fig.update_traces(mode = 'markers')
    fig.show()
    

    return fig.show()



 


the_song = from_song()
df_two = lyrics(the_song[0],the_song[1])[0]
df_two = apply_cleaner(df_two)
tokens = get_tokens(df_two)

artist_in = lyrics(the_song[0],the_song[1])[1]


model = train_model(tokens) 
vect = vector_Collector(model,df_two)
print(df_two)
pred = prediction_matrix(vect[0],model,vect[1])
rel = related_artist_matrix(artist_in,pred,model)
visualization_multi(rel)

print(pd.read_hdf("./RYM-ML-main/Genuis-API/the_df.h5"))

##done

