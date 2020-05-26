# data loading, analysis and wrangling
import pandas as pd  
import numpy as np
from bs4 import BeautifulSoup  # for website parsing and scraping (rotten tomatoes)
import requests  
import html5lib
import re  
import string
import pickle
import os
from tqdm import tqdm

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)

# visualization
import seaborn as sns 
import matplotlib.pyplot as plt

# machine learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

#NLP - sentiment analysis
from textblob import TextBlob


#User defined functions
def text_cleaning(input_text):
    """ remove string punctuation from scapped text """
    input_text = input_text.replace('User Ratings','')
    cleaned_text = "".join([c for c in input_text.strip() if c not in string.punctuation])
    return cleaned_text

def process_movie_name(movie_name):
    """ process movie names to get search string for rottentomato movie review search """
    movie_name = text_cleaning(movie_name)
    processed_movie_name="_".join([str(word).lower() for word in movie_name.split(" ")])
    return processed_movie_name

def process_actor_name(movie_name):
    """ process actor names to get search string for rottentomato actor search """
    movie_name = text_cleaning(movie_name)
    processed_movie_name="_".join([str(word).lower() for word in movie_name.split(" ")])
    return processed_movie_name
    
def get_actor_movies(actor_name): 
    """ scrap all the movie data for specific actor from rottentomato """
    base_url = "https://www.rottentomatoes.com/celebrity/"
    url = base_url + process_actor_name(actor_name)
    r = requests.get(url)    
    soup = BeautifulSoup(r.text, 'html.parser')
    l = []
    try : 
        tbody=soup.find('tbody', attrs={'class':"celebrity-filmography__tbody"})
        table_rows = tbody.find_all('tr')
        
    
        for tr in table_rows:
            td = tr.find_all('td')
            row = [str(tr.text).strip() for tr in td]
            if len(row) == 5:
                l.append(row)
    except : 
        pass        
    movie_list=pd.DataFrame(l, columns=["RATING", "TITLE", "CREDIT", "BOX_OFFICE", "YEAR"])
    return movie_list

def get_movie_reviews(movie_name):
    """ scrap all the available movie review for specific actor from rottentomato """
    base_url = "https://www.rottentomatoes.com/m/"
    url = base_url + process_movie_name(movie_name) + '/reviews'
    r = requests.get(url)    
    soup = BeautifulSoup(r.text, 'html.parser')
    critic_score=soup.find_all('div', attrs={'class' : "row review_table_row"})
    rev=[]
    for row in critic_score:
        rev.append(text_cleaning(row.find('div', attrs={'class' : "the_review"}).text))
    return rev

def get_sentiment_score(actor):
    """ use all the reviews scrapped using get_movie_reviews for a specific actor.
    sentiment polarity of each review is calculated using Textblob. polarity value is between [-1, 1]
    final score is the average score of all the review's sentiment polarity
    """
    movie_reviews = []
    for movie in get_actor_movies(actor).TITLE:
        if len(get_movie_reviews(movie)) !=0 : 
            movie_reviews.extend(get_movie_reviews(movie))

    sentiment_score=0
    count=0        
    for review in movie_reviews:
        if review:
            sentiment_score+=TextBlob(review).polarity
            count+=1
    if count:
        final_score = np.round(sentiment_score/count,2)
    else : 
        final_score = 'NA'
    return final_score

def get_actor_scores(actor):
    """
    Generate a actor wise row to obtain below values
    ['ACTOR','TOTAL_MOVIES','MOVIES_NOT_SCORED','RATING','BOX_OFFICE','REVIEW_SENTIMENT','MOVIES_PER_YEAR','START_YEAR','END_YEAR']
    """
    movie_df = get_actor_movies(actor)
    indexNames = movie_df[ movie_df['YEAR'] == '' ].index
    # Delete these row indexes from dataFrame
    movie_df = movie_df.drop(indexNames)

    count_no_score, rating, rating_counter, total_box_office, avg_rating = 0 , 0, 0, 0, 0
    for i,row in movie_df.iterrows():
        if str(row.RATING).lower() == 'no score yet':
            count_no_score+=1
        else:
            rating += int(''.join([c for c in str(row.RATING) if c.isdigit()]))
            rating_counter += 1

        if len(str(row.BOX_OFFICE))>1:
            total_box_office += float(re.sub("[KM$]", "", row.BOX_OFFICE)) * ( 10**3 if row.BOX_OFFICE.strip()[-1]== 'K' else 10**6 )
    
    review_sentiment_score=get_sentiment_score(actor)
    movie_df['YEAR'] = movie_df['YEAR'].astype(np.int)
    start_year, end_year = movie_df.YEAR.min(), movie_df.YEAR.max()
    avg_mov_per_yr = movie_df.shape[0] / (movie_df.YEAR.max()-movie_df.YEAR.min() + 1)
    
    if rating_counter:
        avg_rating = np.round(rating / rating_counter, 2)
    
    return actor, movie_df.shape[0], count_no_score, avg_rating, total_box_office, review_sentiment_score, avg_mov_per_yr, start_year, end_year