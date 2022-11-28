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




token = "R5iNvu7MuLy8bUWhAY3DTbxpeAGGqj-83JfN7LO2TT98sZTli52wSHSIX1KjIDWl"

genius = lyricsgenius.Genius(token, timeout=15, retries=3)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True 


## For an inputted artist, 
## save the lyrics of their top 100 songs in order to conduct analysis later. 
##Note that the inputted artists should come from the command line. 


## Idea is we store all of the lyrics as indices in a long list, 
##Later we abstract the indices as individual songs
lyrics_storage = []





def lyric_collector2():
    artist_input = input("Artist: ")
    artist_object = genius.search_artist(artist_input, max_songs=100,per_page=50, sort="popularity", include_features=False)
    for i in artist_object.songs:
        #Look into what type artist_object.song returns
        i.lyrics = i.lyrics.replace("\\"," ")
        lyrics_storage.append(i.lyrics)

    
    return [lyrics_storage,artist_input]
        


##Seperates lyrics into the seperate words, then trains it on the word2vec model
def clean_and_train (list_of_lyrics):

    #list_of_word = [i.split(" ") for i in list_of_lyrics]
    one_list = [i for i in list_of_lyrics]
    sublist = []
    tokenize=[]
    one_list =[one_list[i].split(" ") for i in range(len(one_list))]
    for song in one_list:
        for word_list in song: 
            word_list = word_list.translate(str.maketrans('', '', string.punctuation))
            word_list = word_list.lower()
            if "\n" in word_list:
                print(sublist)
                tokenize.append(sublist)
                sublist=[]

            word_list = word_list.split("\n")
            for i in word_list: 
                sublist.append(i)
            #sublist.append(word_list.replace("\n"," "))     

    print({"toke":tokenize})
    
    model = gensim.models.Word2Vec(tokenize, min_count=1)
    
    
    return [model,tokenize] 



def vector_Collector(model,tokenized):
    unique_words  = set()
    for sentence in tokenized:
        for word in sentence:
            unique_words.add(word)

    unique_words = list(unique_words)



    vocab_hold = []
    for i in unique_words: 
        try: 
            vocab_hold.append({i:model.wv.get_vector(i,norm=True)})
        except KeyError as e: 
            continue

   
    vectors_only = [list(i.values()) for i in vocab_hold]

    return [vocab_hold,vectors_only,unique_words]


def prediction(vector_list,model,unique_words):
    vector_list = [i[0].tolist() for i in vector_list]
    kmeans = KMeans(n_clusters=4).fit(vector_list)
    kmeans.predict(vector_list)
    kmeans.cluster_centers_
    data_embed=TSNE(n_components=2, perplexity=50, verbose=2, method='barnes_hut').fit_transform(vector_list)
    df = pd.DataFrame(data_embed,columns=["x","y"])
    df["Words"] = [list(i.keys())[0] for i in unique_words ]
    df["Cluster group"] = kmeans.labels_
    print(df)
    
    return df 



def visualization (df):
    # model.labels_ is nothing but the predicted clusters i.e y_clusters
    labels = df["Cluster group"]
    fig = px.scatter(df, x='x', y='y', color='Cluster group', size_max=6 , text='Words', opacity=0.7)
    fig.update_traces(mode = 'markers')
    fig.show()
    return fig.show()
 




    
    

##Finds optimal cluster number
##More for dev, not for the user. 
def find_optimal_group(vector): 
    WCSS = []
    for i in range(1,11):
        the_vect = [i[0].tolist() for i in vector]
        model = KMeans(n_clusters = i).fit(the_vect)
        WCSS.append(model.inertia_)
    fig = plt.figure(figsize = (7,7))
    plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
    plt.xticks(np.arange(11))
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()





def related_artist_matrix (artist_input,artist_prediction_matrix):
    lst = get_related_artists(artist_input)[:3]
    lyric_per = []
    starter_dataframe =  artist_prediction_matrix.copy()
    starter_dataframe["Rapper"] = artist_input
    for i in lst:
        artist_object = genius.search_artist(i, max_songs=100,per_page=50, sort="popularity",include_features=False)
        lyric_per=[]
        for w in artist_object.songs:
            #Look into what type artist_object.song returns
            w.lyrics = w.lyrics.replace("\\"," ")
            lyric_per.append(w.lyrics)
        cleaner = clean_and_train(lyric_per)
        temp = vector_Collector(cleaner[0],cleaner[1])
        per_artist_pred =  prediction(temp[1],cleaner[0],temp[0])
        per_artist_pred["Rapper"] = str(i)
        starter_dataframe =  pd.concat([starter_dataframe,per_artist_pred]).reset_index(drop=True)
    
    return starter_dataframe
    

        


def visualization_multi (df):
    # model.labels_ is nothing but the predicted clusters i.e y_clusters

    ## General process. 
    ## Find that related artists, then union each data frame to the current one.
    ## Then you can use fig to visualize it. 
    
    labels = df["Cluster group"]
    fig = px.scatter(df, x='x', y='y', color='Cluster group', size_max=6 ,facet_col="Rapper",text= "Words" , opacity=0.7)
    fig.update_traces(mode = 'markers')
    fig.show()

    return fig.show()
 


lyrics = lyric_collector2()
cleaned = clean_and_train(lyrics[0])
listing = vector_Collector(cleaned[0],cleaned[1])


pred = prediction(listing[1],cleaned[0],listing[0])
mat = related_artist_matrix(lyrics[1],pred)

visualization_multi(mat)
#find_optimal_group(listing[1])
























