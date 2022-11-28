from sklearn.neighbors import NearestNeighbors
from skimage import io
from tabulate import tabulate
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import sys

cid = 'c480b13ef81c4e6aa0ab0119636eabe5'
secret = '50826f24c12044448b906de50ac74742'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Prints the 5 most similar albums
def printSimilar(titles, artists, uris, indices, userAlbumIndex):
    similarAlbumsList = list(indices[userAlbumIndex])  # Specified Album to find similar albums to
    albums = []
    recommendedUris = []
    recommendedTitles = []
    recommendedArtists = []

    for i in range(len(similarAlbumsList)):
        albums.append([artists[similarAlbumsList[i]], titles[similarAlbumsList[i]], uris[similarAlbumsList[i]]])
        recommendedUris.append(uris[similarAlbumsList[i]])
        recommendedTitles.append(titles[similarAlbumsList[i]])
        recommendedArtists.append(artists[similarAlbumsList[i]])

    print(tabulate(albums, headers=['Artist', 'Title', 'URI']))
    visualizeAlbums(recommendedUris, recommendedTitles, recommendedArtists)

# Plot similar albums
def visualizeAlbums(uris, titles, artists):
    urls = []  # List of cover art URLS

    for uri in uris:
        result = sp.album(uri)
        urls.append(result['images'][0]['url'])  # Append cover art URL to list of image URLS

    plt.figure(figsize=(30, int(.8 * len(uris))), facecolor='#ffeba3')
    columns = len(urls)

    for i, url in enumerate(urls):
        plt.subplot(int(len(urls) / columns), columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        s = ''
        plt.xlabel(s.join(titles[i] + '\n' + artists[i]), fontsize=8, fontweight='bold')
        plt.tight_layout(h_pad=15, w_pad=10)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()

# Using unsupervised KNN to get similar albums (euclidean distance)
def recommend(albumsDataframe):
    # Get user album index
    userAlbumIndex = None
    emptyIndex = True
    while emptyIndex:
        albumInput = input("Album title: ")
        searchResult = albumsDataframe[albumsDataframe['Title'].str.contains(albumInput, na=False)]

        emptyIndex = searchResult.empty

        if not emptyIndex:
            userAlbumIndex = searchResult.index.tolist()[0]

    # Normalize columns with un-normalized values
    albumsDataframe[['key', 'loudness', 'tempo', 'duration_ms', 'time_signature']] = (albumsDataframe[
                                                                                          ['key', 'loudness', 'tempo',
                                                                                           'duration_ms',
                                                                                           'time_signature']] -
                                                                                      albumsDataframe[
                                                                                          ['key', 'loudness', 'tempo',
                                                                                           'duration_ms',
                                                                                           'time_signature']].min()) / (
                                                                                                 albumsDataframe[
                                                                                                     ['key', 'loudness',
                                                                                                      'tempo',
                                                                                                      'duration_ms',
                                                                                                      'time_signature']].max() -
                                                                                                 albumsDataframe[
                                                                                                     ['key', 'loudness',
                                                                                                      'tempo',
                                                                                                      'duration_ms',
                                                                                                      'time_signature']].min())

    # albumDescriptors[[]] =

    # Selecting audio features for KNN
    #albumValues = albumsDataframe[
        # ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        #  'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']]

    albumValues = albumsDataframe.drop(['Title', 'Artist', 'URI', 'Descriptor Count', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'], axis = 1)
    albumValuesCols = albumValues.columns.tolist()
    albumsDataframe[albumValuesCols] = albumValues[albumValuesCols].apply(lambda x: x / 5.5)  # Weighting of descriptors (higher we divide by, less weight)

    #print("Max Descriptor Count", albumsDataframe['Descriptor Count'].max())

    albumValues = albumsDataframe.drop(['Title', 'Artist', 'URI', 'Descriptor Count'], axis=1)

    # energy, key, mode, speechiness, liveness, tempo, duration_ms, time_signature important
    # acousticness, instrumentalness, valence, loudness maybe important
    # acousticness, instrumentalness,  might work

    # Next step is to improve accuracy (potentially use genres)
    # albumValues = albumsDataframe[['loudness', 'energy', 'key', 'mode', 'speechiness', 'liveness', 'tempo', 'duration_ms', 'time_signature']]

    albumTitles = list(albumsDataframe['Title'])  # List of album titles
    albumArtists = list(albumsDataframe['Artist'])  # List of album artists
    albumURIs = list(albumsDataframe['URI'])  # List of album artists

    albumValues = albumValues.to_numpy(dtype=np.float32)  # np array of all features for albums
    similarAlbums = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(albumValues)
    distances, indices = similarAlbums.kneighbors(albumValues)

    print(distances[userAlbumIndex])  # For debugging

    printSimilar(albumTitles, albumArtists, albumURIs, indices, userAlbumIndex)  # Print out similar albums


def combineSpotifyWithDescriptors(spotifyPath, descriptorPath):
    descriptorPath.reset_index(drop=True, inplace=True)  # Reset indices

    newAlbumDataframe = pd.concat([spotifyPath, descriptorPath], axis=1, join="inner")  # Join spotify and descriptor data

    newAlbumDataframe = removeBadRows(newAlbumDataframe)

    return newAlbumDataframe

def removeBadRows(df):
    # Remove rows with no descriptors
    indices = df.loc[df['Descriptor Count'] == 0].index
    df.drop(indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove rows with no spotify audio features (sum of 0)
    # This should not happen but just in case
    indexNames = df[((df['danceability'] + df['energy'] + df['key'] + df['loudness'] + df['mode'] + df['speechiness'] + df['acousticness'] + df['instrumentalness'] + df['liveness'] + df['valence'] + df['tempo'] + df['duration_ms'] + df['time_signature']) == 0)].index
    df.drop(indexNames, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    # spotifyData = pd.read_pickle('Spotify API Connection/albums_audio_features.pkl')
    # descriptorData = pd.read_pickle("Recommender/descriptors_data_priori_-4415.pkl")
    #
    # all_data = combineSpotifyWithDescriptors(spotifyData, descriptorData)
    # all_data.to_csv('Recommender/all_data.csv')

    all_data = pd.read_pickle('Recommender/all_data.pkl')
    recommend(all_data)

    # Next step is to have a weighting of descriptors for each album by bins potentially or just by ordering


if __name__ == "__main__":
    main()