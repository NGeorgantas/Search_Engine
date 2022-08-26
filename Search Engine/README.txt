Final.py is our search engine

In order to run Final.py you have to change the working directory to the location that the folder Final.py is in your PC.

To do that, change the path in the 2nd line of the code with the command os.chdir(r"~\Search Engine and Web Mining Final\Search Engine and Web Mining Final\Search Engine")

It read our dataset from sample_100_without_duplicates.json
Corrects misspelled words with the use of english_dictionary.json and group_words.json
Expands the query with the use of w2v, file model.model