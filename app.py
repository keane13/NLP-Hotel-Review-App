#Necessary imports
import streamlit as st
import pandas as pd
from   matplotlib import pyplot as plt
import numpy as np
import re
import string
import pickle
import tensorflow as tf
import keras 



#model = load_model("C:\Users\USER\Desktop\macth2bangalore\project\Plant_Disease_Flask_App\Hotel_Review\model_hotel.pkl")


#Headings for Web Application



#Picking what NLP task you want to do
 #option is stored in this variable
#CLASS_NAMES = ['Positive Review', 'Bad Review']
#Textbox for text user is entering


#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type

punctuations = re.sub(r"[!<_>#:)\.]", "", string.punctuation)

def punct2wspace(text):
    return re.sub(r"[{}]+".format(punctuations), " ", text)

def normalize_wspace(text):
    return re.sub(r"\s+", " ", text)

def casefolding(text):
    return text.lower()

def predict(text):
     #model = pickle.load(open('C:/Users/USER/Desktop/Desktop/macth2bangalore/project/Hotel_review/model_hotels.pkl', 'rb'))
     model = tf.keras.models.load_model('hotel_model')
     text = punct2wspace(text)
     text = normalize_wspace(text)
     text = casefolding(text)
    # max_features = 10000      # Jumlah kosakata
    # embedding_dim = 16
     #encoder = keras.layers.TextVectorization(max_tokens=max_features)
    # Latih tokenizer pada data teks
    #Plotting sentiment scores per sentencein line graph
     predictions = model.predict(text)
     return predictions

st.title('Hotel Review Sentiment Analyzer')
text = st.text_area('Tulis Review hotel','..')
if st.button('Analyze'):
    if text is not None:
      with st.spinner('Analyzing the text …'):
       prediction=predict(text)
       if prediction > 0.6:
          st.success('Positive review with {:.2f} confidence'.format(prediction))
          st.balloons()
       elif prediction <0.4:
          st.error('Negative review with {:.2f} confidence'.format(1-prediction))
else:
     st.warning('Coba tulis Review lebih banyak')

    #Polarity and Subjectivity of the entire text inputted
    #sentimentTotal = entireText.sentiment
    #st.write("The sentiment of the overall text below.")
    #st.write(sentimentTotal)


   