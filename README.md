# Lyrics_project
Analyzing the similarity of songs based on lyric usage. 


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






