In order to run Clustering.py and SVD.py you have to change the working directory to the location that the folder these scripts are in your PC.

To do that, change the path in the 18th line of the code with the command os.chdir(r"~\Search Engine and Web Mining Final\Search Engine and Web Mining Final\Clustering")

and in the 2nd line for SVD.py

CLustering.py is an attempt to cluster the documents and label them with relevant keywords

SVD.py is our search engine and the query fetches most relevant documents after clustering


Files: 

english_dictionary.json and group_words.json are used for the spell checking we have

sample_100_without_duplicates.json is our dataset without duplicates

vector csv is a big matrix with tf-idf values, instead of trying to build the matrix in the SVD file we made it once and we load it from the .csv 
in order to make the program faster

Num_dic_data.json is every document we have with and id in order to find the most relevant document after the SVD