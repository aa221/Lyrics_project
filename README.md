# Similar Songs Based on their Lyrics!
Creating a model to predict similar songs based on lyric usage. 


## Context behind Word2Vec and Doc2Vec
Doc2Vec is a generalization of the NLP model Word2Vec. At a high level, Word2Vec uses a Neural Network to learn associations 
between words within large corpus' of texts. Associations are built by looking at surrounding words for each word. In other words, 
if two words are used in the same contexts then these two words are related. For example: 

* The boy slept because he was tired
* The girl slept because she was tired

Here, boy and girl will be closely associated as the surrounding words are the same. 

**Therefore synonyms are very closely related**

Note to create these associations, each words is represented by a vector. The closer two vectors are in distance, the more similar the words they represent are. 

Doc2Vec therefore, is a generalization of Word2Vec, where each document is assigned a vector. In a sense Doc2Vec takes the average of the word vectors, within a document to create a generalized vector. 

Overall the logic then between how documents are related follows the same as for how words are related. 


## Product flow

* User inputs a song and artist.
* If the artist was previously searched for then the wait time will range between 5 and 7 min. 
* If the artist was not searched for then the wait time can reach 20 min. 
* A visualization is shown to the user showing them similar songs and artists to the inputed artist and song. 

## Visualization breakdown 

Below is a representation of the final output shown to a user who inputed Kanye West as the Artist and Through the Wire as the Song. 

<img width="1346" alt="Screen Shot 2022-11-27 at 4 12 21 PM" src="https://user-images.githubusercontent.com/57921290/204177387-8cedcea5-7170-4cea-8edb-b0ba28a2ffbf.png">

* Kid Cudi, Pusha T and Big Sean are all similar artists (deemed by the spotify API).
* Thus 50 songs per related artist (parameter can be changed) are shown on the 2D visualizer.
* The closer two songs are, the more similar they are.
* Users can hover over each song and understand which songs are similar to not only their inputted song, but songs by their inputted artist and related artists.

## Technical Breakdown

* The Spotify API is identify related artists to a user's inputted artist. 
* The Genuis API is used to scrape the lyrics of the top 50 songs from the inputted artist and related ones. 
* Each song's lyrics is assigned a vector, through the Doc2Vec tool. 
* K-means is then ran on all songs across all related artists. 
* TSNE is used to reduce the dimensions of the vectors and project them onto a 2D plane for visualization purposes!


## Next steps

Add spotify features so that similar songs are also similar by their instrumentals and not just lyrics (as it currently stands).








