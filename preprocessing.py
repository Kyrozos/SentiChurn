import numpy as np
import pandas as pd
import tensorflow as tf
from mongo_connection import db_sa
from model_loader import tokenizer_sentiment, label_encoder_churn

def preprocess_churn_data(categorical_features):
    """
    This function preprocesses churn data from the MongoDB collection.

    Steps:
    1. Access the 'churn' collection in the database.
    2. Convert the collection data to a Pandas DataFrame.
    3. Drop the unnecessary '_id' column.
    4. Replace empty spaces with `np.nan` (not a number) and remove rows with missing values.
    5. Extract product IDs and drop the 'ProdID' column from the DataFrame.
    6. Encode categorical features using pre-existing label encodings.
    7. Convert DataFrame to float type.
    8. Return the preprocessed DataFrame and product IDs.
    """   
    churn_collection = db_sa['churn']
    df_churn = pd.DataFrame(list(churn_collection.find()))
    df_churn = df_churn.drop(columns=['_id'])
    df_churn = df_churn.replace(' ', np.nan).dropna()

    prod_ids = df_churn['ProdID'].values.astype('int')
    df_churn = df_churn.drop(columns=['ProdID'])
    
    for col in categorical_features:
        df_churn[col] = df_churn[col].map(label_encoder_churn[col])

    df_churn = df_churn.astype('float')
    return df_churn, prod_ids

def preprocess_sentiment_data(max_length=60):
    """
    This function preprocesses sentiment data from the MongoDB collection.

    Steps:
    1. Access the 'reviews' collection in the database.
    2. Convert the collection data to a Pandas DataFrame.
    3. Drop the unnecessary '_id' column.
    4. Replace empty spaces with `np.nan` (not a number) and remove rows with missing values.
    5. Extract product IDs and drop the 'ProdID' column from the DataFrame.
    6. Convert text data to sequences using the sentiment tokenizer.
    7. Pad the sequences to a maximum length (`max_length`) for consistent model input.
    8. Return the preprocessed sequences (X) and product IDs.
    """
    sentiment_collection = db_sa['reviews']
    df_sentiment = pd.DataFrame(list(sentiment_collection.find()))
    df_sentiment = df_sentiment.drop(columns=['_id'])
    df_sentiment = df_sentiment.replace(' ', np.nan).dropna()

    prod_ids = df_sentiment['ProdID'].values.astype('int')
    df_sentiment = df_sentiment.drop(columns=['ProdID'])

    X = tokenizer_sentiment.texts_to_sequences(df_sentiment['Text'])
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_length)
    return X, prod_ids
