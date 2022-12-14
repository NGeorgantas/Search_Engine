import os
os.chdir(r"C:\Users\georg\Downloads\Search Engine and Web Mining Final\Search Engine and Web Mining Final\PrecisionRecall")


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


    scores = [[i, 0] for i in range(len(data))]
    for word, value in processed_q.items():
        for doc in inverted_index[word]['postings_list']:
            index, weight = doc
            scores[index][1] += value * weight

    scores.sort(key=lambda doc: doc[1], reverse=True)
    sum=0
    
    for i in scores:
        sum+=i[1]
    thresh=sum/len(scores)
    #final_res=[list(i[0],i[1]) for i in scores if i[1]>=thresh]
    final_res=[]
    for i in scores:
        if(i[1]>=thresh):
            final_res.append([i[0],i[1]])
    
    print('----- Results ------ ')
    k=0
    for index,score in enumerate(final_res):
        print('{}. {} - {}'.format(index + 1, docs[score[0]]["url"], score[1]))
        if k>5:
            break
        k=k+1   
    return final_res,docs,list(processed_q.keys()),query



path="sample_100_without_duplicates.json"
inv_indx,data,docs=main_1(path)

results,documents,process_q,query=main_2(inv_indx,data,docs)

#results=main_2(inv_indx,data,docs)
# Query1 : nrows alternative for read_sql_table
#Query 2 : incompatible types: JSONLoader cannot be converted to Loader<JSONObject>
#Query3: Laravel eager loading - one to many relation
#Query 4 : python labelling new data points in a histogram
#Query 5: matrices multiplication in mathematica

import json
file="Num_dic_data.json"
loaded_json=json.load(open(file))

#this is for the query results
num_quer=5
path=r"C:\Users\georg\Downloads\Search Engine and Web Mining Final\Search Engine and Web Mining Final\PrecisionRecall"
labelling1={}
for i in results:
    if any(word in loaded_json[str(i[0])][1] for word in process_q):
        labelling1[i[0]]=str(1)
    else:
        labelling1[i[0]]=str(2)

a_file=open(path+f"\Q{num_quer}_results_labels_Recall.json","w")
a_file=json.dump(labelling1,a_file)

#this is for all the data for all data we choose only the most important words from the wuery that we think 
#would actually made the difference to the meaning 
check_q=['matric','mathematica']
labelling2={}
for i in loaded_json:
    if any(word in loaded_json[i][1] for word in check_q):
        labelling2[i]=str(1)
    else:
        labelling2[i]=str(2)

a_file=open(path+f"\Q{num_quer}_all_labels_Recall.json","w")
a_file=json.dump(labelling2,a_file)

b_file1=open(path+f"\Q{num_quer}_all_labels_Recall.json","r")
all_labels=json.load(b_file1)
len(all_labels)

b_file2=open(path+f"\Q{num_quer}_results_labels_Recall.json","r")
results_labels=json.load(b_file2)
len(results_labels)

def precision(qresults,all_labels):
    #precision=tp/(tp+fp)
    true_pos=[i for i in qresults if (all_labels[str(i)]=='1' and qresults[str(i)]=="1")]
    false_pos=[i for i in qresults if (all_labels[str(i)]=='2' and qresults[str(i)]=="1")]
    return len(true_pos)/(len(true_pos)+len(false_pos))

def recall(qresults,all_labels):
    #recall = tp/(tp+fn)
    true_pos=[i for i in qresults if (all_labels[str(i)]=='1' and qresults[str(i)]=="1")]
    false_neg=[i for i in qresults if (all_labels[str(i)]=='1' and qresults[str(i)]=="2")]
    return len(true_pos)/(len(true_pos)+len(false_neg))


Q1_precision=precision(results_labels,all_labels)
Q1_Recall=recall(results_labels,all_labels)
F1=2*Q1_precision*Q1_Recall/(Q1_precision+Q1_Recall)
print(f"For Query : {query}")
print(f"Precision = {Q1_precision}, Recall = {Q1_Recall}, F1 = {F1}")
results_dic={"query":query,"Precision":Q1_precision,"Recall":Q1_Recall,"F1":F1}

b_file3=open(path+f"\Q{num_quer}Precision n Recall Results.json","w")
b_file3=json.dump(results_dic,b_file3)
