import os 
os.chdir(r'C:\Users\georg\Downloads\Search Engine and Web Mining Final\Search Engine and Web Mining Final\Clustering')

def data_reader(file):
    import json
    import pandas as pd
    my_dict={}
    loaded_json=json.load(open(file))
    doc_list=loaded_json["All questions"]
    for i in range(len(doc_list)):
        my_dict[doc_list[i]['url']]=doc_list[i]['title']+" "+doc_list[i]['description']
    return my_dict,doc_list

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
    
def corpus_creation(some_dict):
    from collections import Counter
    corpus=[]
    for i in some_dict.keys():
        words=preprocess_text(some_dict[i])
        bag_of_words = Counter(words)
        corpus.append(bag_of_words)
    return corpus

def compute_idf(corpus):
    from collections import defaultdict
    import math
    num_docs = len(corpus)
    idf = defaultdict(lambda: 0)
    for doc in corpus:
        for word in doc.keys():
            idf[word] += 1

    for word, value in idf.items():
        idf[word] = math.log(num_docs / value)
    return idf

def compute_weights(idf,doc):
    x=max(doc.values())
    for word, value in doc.items():
        doc[word] =  (doc[word]/x) * idf[word]
    
        
def normalize(doc):
    import math
    denominator = math.sqrt(sum([e ** 2 for e in doc.values()]))
    for word, value in doc.items():
        doc[word] = value / denominator
        
def build_inverted_index(idf, corpus):
    inverted_index = {}
    for word, value in idf.items():
        inverted_index[word] = {}
        inverted_index[word]['idf'] = value
        inverted_index[word]['postings_list'] = []

    for index, doc in enumerate(corpus):
        for word, value in doc.items():
            inverted_index[word]['postings_list'].append([index, value])

    return inverted_index

def query_preprocess(text,inverted_index):
    from collections import Counter
    words_dictionary=set(inverted_index.keys())
    query=preprocess_text(text)
    query = [word for word in query if word in words_dictionary]
    query = Counter(query)
    return query

def english_dictionaries():
    import nltk
    from nltk.corpus import words
    word_list = words.words()
    english_dictionary={}
    set_dictio=set([])
    group_words={}
    num_dictionary={}
    for word in word_list:
        english_dictionary[word.lower()]=word.lower()
        num_dictionary[word]=len(word)
    for word in word_list:
        set_dictio.add(num_dictionary[word])
    for i in set_dictio:
        group=[]
        for word in word_list:
            if(int(i)==len(word)):
                group.append(word.lower())
        group_words[i]=group
    return english_dictionary,group_words

def minimum_distance_words(similar_words,word_to_check):
    from Levenshtein import distance
    distances=set([])
    check={}
    results=[]
    for i,j in enumerate(similar_words):
        edit_dist=distance(word_to_check,j)
        check[j]=edit_dist
        distances.add(edit_dist)
    for key in check.keys():
        if(check[key]==min(distances)):
            results.append(key)
    return results

def auto_correct2(word,english_dict,word_groups):
    preprocess_L=[]
    words_to_check=[]
    for i in range(len(word)-1,len(word)+2):
        preprocess_L.extend(word_groups[str(i)])#str()
    for checkword in preprocess_L:
        if(word[0]==checkword[0]):
            words_to_check.append(checkword)
    results=minimum_distance_words(words_to_check,word)
    if len(results)>1:
        print(results)
        print("Something is wrong with your input!")
        final_result=int(input("Please choose the index of one of the recomended words from the list or enter 0 to return the original word :"))
        if(final_result==0):
            return word
        else:
            return results[final_result-1]
    elif(len(results)==0):
        return word
    return results[0]

def check_string(my_string,english_dict,word_groups):
    my_string=remove_nonwords(my_string)
    results=[]
    for item in my_string.split():
        try:
            x=english_dict[item.lower()]
        except:
            x=auto_correct2(item.lower(),english_dict,word_groups)
        results.append(x)
    final=" ".join(results)
    return final

def main_1(path):
    data,docs=data_reader(path)
    corpus=corpus_creation(data)
    idf=compute_idf(corpus)
    for doc in corpus:
        compute_weights(idf, doc)
        normalize(doc)
    inverted_index = build_inverted_index(idf, corpus)
    return inverted_index,data,docs

def main_2(inverted_index,data,docs):
    import math    
    import json

    #english_dict,word_groups=english_dictionaries()
    a_file = open("english_dictionary.json", "r")
    english_dict = json.load(a_file)
    b_file = open("group_words.json", "r")
    word_groups = json.load(b_file)

    

    query=input("Give Query: ")

    final_string=check_string(query,english_dict,word_groups)
    print(final_string)
    processed_q=query_preprocess(final_string,inverted_index)
    
    

    
    

    x=max(processed_q.values())

    for word, value in processed_q.items():
        processed_q[word] = inverted_index[word]['idf'] * ((processed_q[word]/x) )
    normalize(processed_q)
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    from kneed import KneeLocator
    vectorframe = pd.read_csv("vector.csv" ,index_col=False)
    vectorframe=vectorframe.drop(columns=['Unnamed: 0','level_0'])
    data2 = vectorframe.fillna(0)
    svd = TruncatedSVD(n_components =100)
    svd.fit(data2)
    transformed = svd.transform(data2)
    svd_frame =pd.DataFrame(transformed)
    #============================SVD_to_query=====================================================#
    n = 10  # every 100th line = 1% of the lines
    df1 = pd.read_csv('vector.csv', header=0, skiprows=lambda i: i % n != 0)
    data3 = df1.fillna(0)
    query_df=pd.DataFrame.from_dict(processed_q,orient='index')
    query2=query_df.T
    frames = [data3, query2]
    result = pd.concat(frames)
    result=result.fillna(0)
    result

    svd.fit(result)
    transformed_query = svd.transform(result)
    svd_query =pd.DataFrame(transformed_query)
    svd_query=pd.DataFrame(transformed_query).iloc[-1]
    kmeans_kwargs = {"init":"k-means++", "n_init":11, "max_iter": 300, "random_state":42}
    sse =[]
    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeans.fit(svd_frame)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1,11),sse, curve = 'convex', direction = "decreasing")

    num = kl.elbow
    kmeans = KMeans(n_clusters=14, init="k-means++",max_iter=300,n_init = 11)
    kmeans.fit(svd_frame)
    lowestSSE = kmeans.inertia_
    centroids = kmeans.cluster_centers_
    label = kmeans.fit_predict(svd_frame)
    from sklearn.metrics.pairwise import cosine_similarity
    distance_matrix = cosine_similarity([svd_query],centroids)
    u=np.argsort(distance_matrix)
    u=u[0][-1]
    distance_matrix2 = cosine_similarity([svd_query],svd_frame[label==u])
    x=np.argsort(distance_matrix2[0])
    x=x[-1]

    docu=svd_frame[label==u].index
    final=docu[x]
    import json
    file_dictionary=open('Num_dic_data.json',mode='r')
    loaded_json=json.load(file_dictionary)
    
    return loaded_json[str(final)][1]

path="sample_100_without_duplicates.json"
inv_indx,data,docs=main_1(path)

theta=main_2(inv_indx,data,docs)
print(theta)