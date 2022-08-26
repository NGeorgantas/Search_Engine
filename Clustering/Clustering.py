import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json

import os
os.chdir(r'C:\Users\georg\Downloads\Search Engine and Web Mining Final\Search Engine and Web Mining Final\Clustering')

def remove_nonwords(text):
    non_words = re.compile(r"[^a-z']")
    processed_text = re.sub(non_words, ' ', text)
    return processed_text.strip()

def remove_stopwords(text):    
    stopwrds=stopwords.words('english')
    words = [word for word in text.split() if word not in stopwrds]
    return words

def stem_words(words):   
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

def preprocess_text(corpus):
    final_corpus=[]
    for text in corpus:
        processed_text = remove_nonwords(text.lower())
        words = remove_stopwords(processed_text)
        stemmed_words = stem_words(words)
        final_corpus.append(" ".join(stemmed_words))
    return final_corpus

def query_process(query):
    pr_q=remove_nonwords(query)
    pr_q_stp=remove_stopwords(pr_q)
    q_stem=stem_words(pr_q_stp)
    return " ".join(q_stem)

def print_related_docs(doc_vec,query_vec,corpus):
    cosineSim=cosine_similarity(doc_vector,query_vector).flatten()
    related_docs_inx=cosineSim.argsort()[:-len(corpus)-1:-1]
    print(f"Related documents per order : {related_docs_inx}")
    print(" ")
    print("TOP 5 RESULTS")
    print(" ")
    for i in range(5):
        print(corpus[related_docs_inx[i]],'\n')


data = pd.read_json('data_dic_data.json', lines=True)
data=data.T
data=data.rename(columns={0:"contents"})

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(data.contents)
text = tfidf.transform(data.contents)

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(text, 20)

clusters = MiniBatchKMeans(n_clusters=5, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()
    
plot_tsne_pca(text, clusters)

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)

"""file="data_dic_data.json"
loaded_json=json.load(open(file))
pre_processed_corpus=[loaded_json[description] for description in loaded_json]
processed_corpus=preprocess_text(pre_processed_corpus)
vectorizerX=TfidfVectorizer()
vectorizerX.fit(processed_corpus)
doc_vector=vectorizerX.transform(processed_corpus)

user_query=input("Please give a query: ")
processed_query=[query_process(user_query)]
query_vector=vectorizerX.transform(processed_query)
print_related_docs(doc_vector,query_vector,pre_processed_corpus)"""