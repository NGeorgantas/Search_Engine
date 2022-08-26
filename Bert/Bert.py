from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import json
import numpy as np
import pandas as pd
import re
import os

os.chdir(r'C:\Users\georg\Downloads\Search Engine and Web Mining Final\Search Engine and Web Mining Final\Bert')
#query="Python labelling new data points in a histogram!!!"

def remove_nonwords(text):
    non_words = re.compile(r"[^a-zA-Z']")
    processed_text = re.sub(non_words, ' ', text)
    return processed_text.strip()

def data_reader(file):
    my_dict={}
    num_dict={}
    loaded_json=json.load(open(file))
    doc_list=loaded_json["All questions"]
    for i in range(len(doc_list)):
        my_dict[doc_list[i]['url']]=remove_nonwords(doc_list[i]['title']+" "+doc_list[i]['description'])
        num_dict[i]=[doc_list[i]['url'],remove_nonwords(doc_list[i]['title']+" "+doc_list[i]['description'])]
    return my_dict,doc_list,num_dict

def present_results(top_results,documents,cos_sim,stop=0):
    res=[]
    if stop==0:
        stop=len(top_results)
    print("----- Results ------")
    for i,j in enumerate(top_results):
        if i>stop:
            break
        print(i+1,".",documents[j]['url'],"-",cos_sim[j])
        res.append((documents[j]['url'],cos_sim[j]))
    return res

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

path="sample_100_without_duplicates.json"
data,docs,num_docs=data_reader(path)

b_file=open("BERT_embeddings_example_assignment2.json", "r")
brt_emb=json.load(b_file)
embeddings_2=[np.asarray(brt_emb[i]) for i in  brt_emb]

#query="Python labelling new data points in a histogram!!!"
a_file = open("english_dictionary.json", "r")
english_dict = json.load(a_file)
b_file = open("group_words.json", "r")
word_groups = json.load(b_file)
query=input("Give your query :")
query=remove_nonwords(query)
final_string=check_string(query,english_dict,word_groups)
print(" ")
print(f"Fetching data for :{final_string}")
print(" ")
model = SentenceTransformer('bert-base-nli-mean-tokens')
query_embeddings=model.encode([query])
results=cosine_similarity(query_embeddings,embeddings_2)
fin_res=results[0]
cos_sims=np.argsort(fin_res)[::-1]
top=list(cos_sims)
final_res=present_results(top,docs,fin_res)

