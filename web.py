import streamlit as st
import pandas as pd
import numpy as np
import pickle

loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

st.title('Predictor de Objetivo o Subjetivo')

st.write('Este predictor clasifica si un documento es objetivo o subjetivo.')

totalWordsCount = st.number_input('Total number of words in the article')
semanticobjscore = st.number_input('Frequency of words with an objective SENTIWORDNET score')
semanticsubjscore = st.number_input('Frequency of words with a subjective SENTIWORDNET score')
CC = st.number_input('Frequency of coordinating conjunctions')
CD = st.number_input('Frequency of numerals and cardinals')
DT = st.number_input('Frequency of determiners')
EX = st.number_input('Frequency of existential there')
FW = st.number_input('Frequency of foreign words')
INs = st.number_input('Frequency of subordinating preposition or conjunction')
JJ = st.number_input('Frequency of ordinal adjectives or numerals')
JJR = st.number_input('Frequency of comparative adjectives')
JJS = st.number_input('Frequency of superlative adjectives')
LS = st.number_input('Frequency of list item markers')
MD = st.number_input('Frequency of modal auxiliaries')
NN = st.number_input('Frequency of singular common nouns')
NNP = st.number_input('Frequency of singular proper nouns')
NNPS = st.number_input('Frequency of plural proper nouns')
NNS = st.number_input('Frequency of plural common nouns')
PDT = st.number_input('Frequency of pre-determiners')
POS = st.number_input('Frequency of genitive markers')
PRP = st.number_input('Frequency of personal pronouns')
PRPS = st.number_input('Frequency of possessive pronouns')
RB = st.number_input('Frequency of adverbs')
RBR = st.number_input('Frequency of comparative adverbs')
RBS = st.number_input('Frequency of superlative adverbs')
RP = st.number_input('Frequency of particles')
SYM = st.number_input('Frequency of symbols')
TOs = st.number_input("Frequency of 'to' as preposition or infinitive marker")
UH = st.number_input('Frequency of interjections')
VB = st.number_input('Frequency of base form verbs')
VBD = st.number_input('Frequency of past tense verbs')
VBG = st.number_input('Frequency of present participle or gerund verbs')
VBN = st.number_input('Frequency of past participle verbs')
VBP = st.number_input('Frequency of present tense verbs with plural 3rd person subjects')
VBZ = st.number_input('Frequency of present tense verbs with singular 3rd person subjects')
WDT = st.number_input('Frequency of WH-determiners')
WP = st.number_input('Frequency of WH-pronouns')
WPS = st.number_input('Frequency of possessive WH-pronouns')
WRB = st.number_input('Frequency of WH-adverbs')
baseform = st.number_input('Frequency of infinitive verbs')
Quotes = st.number_input('Frequency of quotation pairs in the entire article')
questionmarks = st.number_input('Frequency of question marks in the entire article')
exclamationmarks = st.number_input('Frequency of exclamation marks in the entire article')
fullstops = st.number_input('Frequency of full stops')
commas = st.number_input('Frequency of commas')
semicolon = st.number_input('Frequency of semicolons')
colon = st.number_input('Frequency of colons')
ellipsis = st.number_input('Frequency of ellipsis')
pronouns1st = st.number_input('Frequency of first person pronouns')
pronouns2nd = st.number_input('Frequency of second person pronouns')
pronouns3rd = st.number_input('Frequency of third person pronouns')
compsupadjadv = st.number_input('Frequency of comparative and superlative adjectives and adverbs')
past = st.number_input('Frequency of past tense verbs with 1st and 2nd person pronouns')
imperative = st.number_input('Frequency of imperative verbs')
present3rd = st.number_input('Frequency of present tense verbs with 3rd person pronouns')
present1st2nd = st.number_input('Frequency of present tense verbs with 1st and 2nd person pronouns')
sentence1st = st.number_input('First sentence class')
sentencelast = st.number_input('Last sentence class')
txtcomplexity = st.number_input('Text complexity score')

# Hacer la predicción
if st.button('Predecir'):
    input_data = [[totalWordsCount, semanticobjscore, semanticsubjscore, CC,CD,DT,EX,FW,INs,JJ, JJR, JJS, LS,MD,NN,NNP, NNPS,NNS,PDT,POS,PRP,PRPS,
                   RB,RBR,RBS,RP, SYM, TOs,UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WPS,WRB, baseform, Quotes, questionmarks, exclamationmarks,
                   fullstops, commas, semicolon, colon, ellipsis, pronouns1st, pronouns2nd, pronouns3rd, compsupadjadv, past,
                   imperative, present3rd, present1st2nd, sentence1st, sentencelast,txtcomplexity]]  # Agrega otros parámetros aquí
    input_data_as_numpy_array = np.asarray(input_data) 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)

    st.write('The document is: '+prediction)
