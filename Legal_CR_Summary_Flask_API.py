from flask import Flask
from flask import request

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash


import re
import os
import time
import tqdm
import pandas as pd 
import numpy as np
from tqdm import tqdm
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spacy import displacy
import statistics
from collections import Counter
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")




app = Flask(__name__)
auth = HTTPBasicAuth()
users = {
    "abhinbt@gmail.com": generate_password_hash("flaskapicred"),
    "susan": generate_password_hash("Credentials")
}


@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username



@app.route('/Legal_Case_report_Summary/', methods=['GET'])
@auth.login_required
def Legal_Case_report_summary():
    # Summarize function

    def summarization(coefficients, sentence_num, chrologically, keyphrases):
        print('Summarization process started.')
        processed_df = Namer_entity_Recog()
        calculate_tfidf_scores(processed_df)
        summary = append_finale_score(processed_df, coefficients, sentence_num, chrologically, keyphrases)
        #create_summary_files(summary, file_path, output_dir)
        print('Summarization proces completed.')
    
        return summary

    # Funciton to find Name entity recognition

    def Namer_entity_Recog():
        print('\tNER started.')
        dataframe = preprocessing()
        processed_df = processing(dataframe)
        print("\tDone NER.")
        return processed_df

    def preprocessing():
        print('\t\tPreprocessing started.')
        
        data = []
        text = ""

        text = request.data.decode()

        
        # Cleaning the xml data.
        if(text != ''):
            text=text.strip().replace('\n', '')
            sentences = re.findall(r'<sentence .*?>(.*?)</sentence>+', text)
        
            name = re.findall(r'<name>(.*?)</name>+', text)[0]
        
            data.append([name,sentences])
        
            text = ""
            data = []
            data.append(name)
        
            for sent in sentences:
                if(len(sent) > 10):
                    data.append(sent)
                text += sent + "\n"
            
        df = pd.DataFrame(data, columns = ['Value'])

        print('\t\tDone preprocessing.')
        return df


    def processing(dataframe):
        print('\t\tProcessing started.')
        validator = {'SCORE':0, 'SUM':0, 'TFIDF_SCORE':0, 'PERSON':0,'NORP':0,'FAC':0,'ORG':0,'GPE':0,'LOC':0,'PRODUCT':0,'EVENT':0,'WORK_OF_ART':0,'LAW':0,'LANGUAGE':0,'DATE':0,'TIME':0,'PERCENT':0,'MONEY':0,'QUANTITY':0,'ORDINAL':0,'CARDINAL':0,'SENTENCE_LENGTH':0,'SENTENCE':''}
        document = []
        cnt = 1
        #Looping through the dataframe
        for i in range(1, len(dataframe)):
            # Get sentence from the dataframe
            sentence = dataframe.iloc[i].Value
            #Convert the sentece to nlp format.
            nlp_sentence = nlp(sentence)
            #Key to extract the cardinal score
            keys = {'SCORE':0, 'SUM':0, 'TFIDF_SCORE':0, 'PERSON':0,'NORP':0,'FAC':0,'ORG':0,'GPE':0,'LOC':0,'PRODUCT':0,'EVENT':0,'WORK_OF_ART':0,'LAW':0,'LANGUAGE':0,'DATE':0,'TIME':0,'PERCENT':0,'MONEY':0,'QUANTITY':0,'ORDINAL':0,'CARDINAL':0,'SENTENCE_LENGTH':0,'SENTENCE':''}
            cnt = Counter([x.label_ for x in nlp_sentence.ents])
            print(cnt)
            for k in cnt.keys():
                keys[k] = cnt[k]
            #Check the key cardinala values are not zero.
            if validator != keys:
                keys['SENTENCE'] = sentence
                suma = 0
                for k in keys.keys():
                    if(k != 'SENTENCE'):
                        suma += keys[k]
                keys['SUM'] = suma
                document.append(keys)
                print(document)

        processed_df = pd.DataFrame(document, columns = ['SCORE','SUM','TFIDF_SCORE','PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL','SENTENCE_LENGTH','SENTENCE'])
        print('\t\tDone processing.')
        return processed_df

    def stem_sentence(sentence, stemmer):
        words = nltk.word_tokenize(sentence)
        stemmed_sentence = ''

        for word in words:
            stemmed_sentence += stemmer.stem(word) + ' '

        return stemmed_sentence

    def calculate_tfidf_scores(dataframe):        
        # stemming
        sentences = dataframe['SENTENCE']
        stemmed_sentences = []
        stemmer = PorterStemmer()
        for sentence in sentences:
            stemmed_sentences.append(stem_sentence(sentence, stemmer))

        # Removing stop words
        stopwords = nltk.corpus.stopwords.words('english')
        
        # tf-idf score for each sentence
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        text_vector = vectorizer.fit_transform(stemmed_sentences)
        
        sentence_lengths = []
        sentence_scores = []
        for row in text_vector:
            sentence_length = row.getnnz()            
            if sentence_length == 0:
                sentence_length = 1
            sentence_lengths.append(sentence_length)
            sentence_scores.append(np.sum(row))
        
        # create new column in dataframe with tf-idf scores for each sentence
        dataframe['SENTENCE_LENGTH'] = sentence_lengths      
        dataframe['TFIDF_SCORE'] = sentence_scores
        
        print('\tDone calculating TF-IDF score.')

    def calculate_score(keyphrases, sentence, tfidf_score, sentence_length, dates_num, person, norp, org, money, gpe, law, work_of_art, st_dev, coefficients):
        phrase_score = 0
        sentence = sentence.lower()
        if any(phrase in sentence for phrase in keyphrases):
            phrase_score = 1
        
        max_date = 0.4
        max_person = 0.4
        max_entity = 0.5
        max_phrases = 1
        date_coef = coefficients[0] * max_date
        people_coef = coefficients[1] * max_person
        entity_coef = coefficients[2] * max_entity
        phrases_coef = coefficients[3] * max_phrases
        
        normalized_tfidf = tfidf_score / sentence_length
        peoples_num = person + work_of_art
        entities_num = norp + org + money + gpe + law + person + work_of_art
        
        return normalized_tfidf + st_dev * (date_coef * dates_num + people_coef * peoples_num + entity_coef * entities_num + phrases_coef * phrase_score)


    def append_finale_score(dataframe, coefficients, sentence_num, chrologically, keyphrases):        
        st_dev = statistics.stdev(dataframe['TFIDF_SCORE'])
        dataframe['SCORE'] = dataframe.apply(lambda row: calculate_score(keyphrases, row['SENTENCE'], row['TFIDF_SCORE'], row['SENTENCE_LENGTH'], row['DATE'], row['PERSON'], row['NORP'], row['ORG'], row['MONEY'], row['GPE'], row['LAW'], row['WORK_OF_ART'], st_dev, coefficients), axis=1)
        original_sentences = get_raw_sentences(dataframe)

        dataframe.sort_values('SCORE', inplace=True, ascending=False)
        
        if(sentence_num > len(dataframe)):
            sentence_num = dataframe - 1
            
        #print('\n\n'.join(dataframe['SENTENCE'][0:sentence_num]))
        print(dataframe['SCORE'])
        summary = []
        if(chrologically):
            for i in range(0, len(original_sentences)):                
            
                if original_sentences.loc[i, 'SENTENCE'] in list(dataframe['SENTENCE'])[0:sentence_num]:
                    print(original_sentences.loc[i, 'SCORE'])
                    pair = (original_sentences.loc[i, 'SENTENCE'], original_sentences.loc[i, 'SCORE'])
                    summary.append(pair)
        else:
            for sentence, score in zip(dataframe['SENTENCE'][0:sentence_num], dataframe['SCORE'][0:sentence_num]):               
                pair = (sentence, score)
                summary.append(pair)
        
        print('\tDone calculating finale score.')
        print('---------------------------------')
        return summary

    def get_raw_sentences(dataframe):     
        return dataframe[['SENTENCE', 'SCORE']]


    def theMainFuntion():
        # importance of different factor.
        coefficients = (1, 1, 1, 1)

        # Number of paragraph in output summary.
        sentence_num = 3 * 3
        keyphrases = ""

        # Summary function        
        summary_pairs = summarization( coefficients, sentence_num, True, keyphrases)
        summary = [pair[0] for pair in summary_pairs]
        scores = [pair[1] for pair in summary_pairs]

        print('\n\n'.join(summary))

        sum_final = '\n\n'.join(summary)

        return sum_final


    return theMainFuntion()

    
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)