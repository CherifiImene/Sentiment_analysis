import pandas as pd
import string
import tensorflow as tf

import re
from nltk.tokenize import word_tokenize
from nltk.stem import 	WordNetLemmatizer
from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))
#------------------------------ Utility functions -------------------------------------#
def create_imdb_csv_dataset(path_to_dir, export_path):
    """ 
    read all data in path_to_dir and create a csv file in the export_path path.
    """
    data = [] 
    columns = ["Review", "Sentiment"]
    
    batch_size = 1
    

    tf_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        path_to_dir, 
        batch_size=batch_size,  
        shuffle=True
    )

    for x,y in tf_dataset:
        
       review = str(x.numpy()[0])[2:-1] # to escape quotes
       sentiment = int(y)
       data.append([review,sentiment])
    
    df = pd.DataFrame(data=data,columns=columns)
    
    df.to_csv(export_path)
    return df

#----------------------------------------------------------------------------------------#
def remove_punctuation(string_obj):
    return string_obj.translate(str.maketrans('', '', string.punctuation))

def word_count(text):
    text_no_punc = remove_punctuation(text)
    words= text_no_punc.split()
    word_count = len(words)
    return word_count

#----------------------------------------------------------------------------------------#

def data_processing(text):
    """""
    clean text b
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub('<br />', '', text) 
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]','', text)
    text_tokens = word_tokenize(text)
    
    # remove stop words
    filtered_text = [w for w in text_tokens if not w in stop_words]
    # apply lemmatization
    transformed_tokens = [wordnet_lemmatizer.lemmatize(token) for token in filtered_text]
    return " ".join(transformed_tokens)
