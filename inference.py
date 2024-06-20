import numpy as np
import pandas as pd
import json
import tensorflow as tf
from model_loader import model_churn, model_sentiment, tokenizer_sentiment, label_encoder_churn
from preprocessing import preprocess_churn_data, preprocess_sentiment_data
from recommendation_plans import recommendation_plans
from mongo_connection import db_sa

def inference_churn():
    """
    This function performs churn prediction on preprocessed churn data.

    Steps:
    1. Call `preprocess_churn_data` to get the preprocessed churn data.
    2. Predict churn probabilities for each data point using the loaded churn model.
    3. Create a DataFrame containing product IDs and predicted churn probabilities.
    4. Access the 'churn_inference' collection in MongoDB.
    5. Drop any existing data in the collection (optional, ensures fresh results).
    6. Insert the prediction results from the DataFrame into the collection.
    """
    df_churn, prod_ids_churn = preprocess_churn_data([
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod'
    ]) 
    prediction = model_churn.predict(df_churn)
    
    df_inf_churn = pd.DataFrame({'ProdID': prod_ids_churn, 'Churn': prediction})
    churn_inference = db_sa['churn_inference']
    churn_inference.drop()
    churn_inference.insert_many(json.loads(df_inf_churn.to_json(orient='records')))

def inference_sentiment():
    """
    This function performs sentiment prediction on preprocessed sentiment data.

    Steps:
    1. Call `preprocess_sentiment_data` to get the preprocessed sentiment data.
    2. Predict sentiment (positive or negative) for each text sequence using the sentiment model.
    3. Convert prediction probabilities to integers (0 or 1) and create a DataFrame with product IDs and predicted sentiment.
    4. Access the 'sentiment_inference' collection in MongoDB.
    5. Drop any existing data in the collection (optional, ensures fresh results).
    6. Insert the prediction results from the DataFrame into the collection.
    """
    X_sentiment, prod_ids_sentiment = preprocess_sentiment_data()
    prediction = model_sentiment.predict(X_sentiment)
    prediction = np.round(prediction).astype('int').squeeze()
    df_inf_sentiment = pd.DataFrame({'ProdID': prod_ids_sentiment, 'Sentiment': prediction})
    sentiment_inference = db_sa['sentiment_inference']
    sentiment_inference.drop()
    sentiment_inference.insert_many(json.loads(df_inf_sentiment.to_json(orient='records')))

def inference_product_score(prod_id, alpha=0.5):
    """
    This function calculates a product score based on churn and sentiment predictions.

    Args:
        prod_id (int): The product ID to analyze.
        alpha (float, optional): Weight factor for churn score in the combined score. Defaults to 0.5.

    Returns:
        dict: A dictionary containing the product score, churn score, sentiment score,
        and recommended actions based on the score.
    """
    data_churn = db_sa['churn_inference']
    data_sentiment = db_sa['sentiment_inference']

    churn = pd.DataFrame(list(data_churn.find({'ProdID': prod_id}))) 
    sentiment = pd.DataFrame(list(data_sentiment.find({'ProdID': prod_id})))

    churn_score = churn['Churn'].values
    sentiment_score = sentiment['Sentiment'].values

    churn_score = np.sum(churn_score) / len(churn_score)
    churn_score = 1 - churn_score

    sentiment_score = np.sum(sentiment_score) / len(sentiment_score)

    product_score = alpha * churn_score + (1 - alpha) * sentiment_score
    product_score = np.round(product_score * 10, 1)

    if product_score <= 4:
        rec_output = recommendation_plans['LOW']
    elif product_score <= 7:
        rec_output = recommendation_plans['MEDIUM']
    else:
        rec_output = recommendation_plans['HIGH']
    rec_output["Score"] = product_score
    rec_output["Churn Score"] = churn_score
    rec_output["Sentiment Score"] = sentiment_score
    return rec_output

def analyze_custom_product(custom_data):
    """
    This function analyzes a custom product based on user-provided data.

    Args:
        custom_data (dict): A dictionary containing details of the custom product.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    text_data = custom_data.pop('Text')  # Extract text data
    # Preprocess the text data for sentiment analysis
    text_sequence = tokenizer_sentiment.texts_to_sequences([text_data])
    text_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=60)

    # Predict sentiment
    sentiment_prediction = model_sentiment.predict(text_sequence)
    sentiment_score = np.round(np.mean(sentiment_prediction), 2)  # Calculate sentiment score
    
    # Preprocess the remaining custom data for churn analysis
    churn_data = pd.DataFrame([custom_data])

    # Encode categorical variables using mapping dictionaries
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']:
        churn_data[col] = churn_data[col].map(label_encoder_churn[col])

    churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
    churn_data = churn_data.fillna(0)
    churn_data = churn_data.astype('float')

    # Predict churn
    churn_prediction = model_churn.predict(churn_data)
    churn_score = np.round(1 - np.mean(churn_prediction), 2)  # Calculate churn score
    
    # Calculate product score
    alpha = 0.5  # Weight factor for churn score
    product_score = np.round(alpha * churn_score + (1 - alpha) * sentiment_score, 1)
    
    # Determine recommended actions based on product score
    if product_score <= 4:
        rec_output = recommendation_plans['LOW']
    elif product_score <= 7:
        rec_output = recommendation_plans['MEDIUM']
    else:
        rec_output = recommendation_plans['HIGH']
    rec_output["Score"] = product_score
    rec_output["Churn Score"] = churn_score
    rec_output["Sentiment Score"] = sentiment_score

    return rec_output
