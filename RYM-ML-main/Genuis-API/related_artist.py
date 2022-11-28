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
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import json 
import webbrowser
import spotipy.util as util 
from json.decoder import JSONDecodeError


SPOTIPY_CLIENT_ID = '9b0a115d14994154b4cefa7489c19393'
SPOTIPY_CLIENT_SECRET = 'b2f9be2b157e480d8b5d35499234ddec'
SPOTIPY_REDIRECT_URI='http://google.com'
SCOPE = "user-top-read"


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE))


def get_related_artists(inputted_artist):
    artist_object = sp.search(inputted_artist,limit=1,type="artist")
    artist_id = artist_object["artists"]["items"][0]["id"]
    list_of_related = sp.artist_related_artists(artist_id)
    name_collecter = []
    for i in range(len(list_of_related["artists"])):
        name_collecter.append(list_of_related["artists"][i]["name"])



    print(name_collecter)
    return name_collecter
    

get_related_artists("Eminem")

    
