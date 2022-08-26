import pandas as pd
import gensim
import json
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
os.chdir(r"C:\Users\Nikos\Downloads\Precision Recall\Precision Recall\word2vec trainer")

def data_reader(file):
    my_dict={}
    my_list_of_comments=[]
    loaded_json=json.load(open(file))
    doc_list=loaded_json["All questions"]
    for i in range(len(doc_list)):
        my_dict[doc_list[i]['url']]=doc_list[i]['title']+" "+doc_list[i]['description']
        my_list_of_comments.append(doc_list[i]['description'])
    return my_dict,my_list_of_comments

def remove_nonwords(text):
    import re
    non_words = re.compile(r"[^a-z']")
    processed_text = re.sub(non_words, ' ', text)
    return processed_text.strip()

def remove_stopwords(text):
    from nltk.corpus import stopwords
    stopwrds=stopwords.words('english')
    words = [word for word in text.split() if word not in stopwrds]
    return words

def stem_words(words):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

def preprocess_text(text):
    processed_text = remove_nonwords(text.lower())
    words = remove_stopwords(processed_text)
    stemmed_words = stem_words(words)
    return stemmed_words

def create_corpus(list_of_sentences):
    results=[]
    for sentence in list_of_sentences:
        processed=preprocess_text(sentence)
        results.append(processed)
    return results

path="sample_100_without_duplicates.json"
my_dict,sentences=data_reader(path)

df=pd.DataFrame.from_dict(my_dict, orient='index', columns=['text'])

L=[]
for i in df['text']:
    x=preprocess_text(i)
    L.append(x)

df['words']=L

from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict
sentences =df['words']


import multiprocessing

from gensim.models import Word2Vec
cores = multiprocessing.cpu_count()

model=Word2Vec(window=5,min_count=5,workers=cores-1)

model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=30)

x=model.wv.most_similar("sql")
print(x)


import numpy as np
import matplotlib.pyplot as plt

 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsnescatterplot(model, word):
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    close_words = model.wv.most_similar([word])

    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    reduc = PCA(n_components=2).fit_transform(arrays)

    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.show()

tsnescatterplot(model, 'sql')