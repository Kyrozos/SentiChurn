import pickle
import pymongo
import json
import numpy as np  
import pandas as pd
import streamlit as st
import tensorflow as tf

# Load sentiment analysis model
model_sentiment = tf.keras.models.load_model('weights/sentiment/sentiment_analysis.h5')
model_sentiment.compile(
                        optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision')
                                ]
                        )

# Load sentiment analysis tokenizer
with open('weights/sentiment/sentiment_analysis.pickle', 'rb') as f:
    tokenizer_sentiment = pickle.load(f)

# Load churn analysis model and label encoder
with open('weights/churn/churn_analysis.pickle', 'rb') as f:
    model_churn = pickle.load(f)

with open('weights/churn/label_encoder.pickle', 'rb') as f:
    label_encoder_churn = pickle.load(f)

# Connect to MongoDB
mongo_url = "mongodb+srv://nammar1025:8CtweEd4g3hWf51z@sentichurn.xwbjfyw.mongodb.net/"
client = pymongo.MongoClient(mongo_url)
db_sa = client['SentimentAnalysis']

# Define recommendation plans based on product score
recommendation_plans = {
                        "LOW" : {
                                "Features" : """Prioritize fixing critical bugs or usability issues identified in negative feedback.
Conduct user research to understand missing features causing frustration.""",
                                "UX" : """Analyze user recordings or heatmaps to identify confusing or clunky user flows.
Simplify the user interface based on user feedback.""",
                                "CS" : """Review common customer support tickets to identify areas where response time or resolution needs improvement.
Implement live chat or self-service options for faster resolution.""",
                                "Pricing" : """Analyze customer churn related to pricing and consider offering more flexible pricing plans.
Review competitor pricing strategies.""",
                                "Marketing" : """Re-evaluate marketing messaging based on negative sentiment to ensure it accurately reflects the product's value proposition.
Consider targeting a different customer segment if the current marketing isn't reaching the right audience."""
},
                        "MEDIUM" : {
                                "Features" : """Implement frequently requested features with high potential impact based on user feedback.
A/B test different feature variations to see what resonates best with users.""",
                                "UX" : """Conduct user interviews or surveys to gather detailed feedback on specific aspects of the user experience.
Prioritize UX improvements based on the severity of user pain points.""",
                                "CS" : """Invest in training customer support representatives to handle complex issues more effectively.
Implement customer satisfaction surveys to track improvement in support interactions.""",
                                "Pricing" : """Offer limited-time discounts or promotions to incentivize user retention.
Consider offering tiered pricing plans with different feature sets.""",
                                "Marketing" : """Analyze customer acquisition data to identify the most effective marketing channels.
Refine your marketing campaigns to target existing user segments more effectively."""
},
                        "HIGH" : {
                                "Features" : """Focus on adding innovative features that enhance the core user experience.
Conduct user research to identify potential new features that address unmet user needs.""",
                                "UX" : """Conduct user research to identify opportunities for further UX optimization and user delight.
A/B test different UI/UX variations to optimize user engagement.""",
                                "CS" : """Implement proactive outreach programs to high-value customers.
Offer self-service knowledge base articles and tutorials.""",
                                "Pricing" : """Consider offering loyalty programs or reward systems to incentivize long-term users.
Monitor competitor pricing strategies and adjust accordingly.""",
                                "Marketing" : """Develop customer referral programs to encourage user acquisition through existing satisfied customers.
Invest in building a strong brand community to foster user loyalty."""
}
}

# Function to preprocess churn data
def preprocess_churn_data(
                        categorical_features = [
                                                'gender',
                                                'Partner',
                                                'Dependents',
                                                'PhoneService',
                                                'MultipleLines',
                                                'InternetService',
                                                'OnlineSecurity',
                                                'OnlineBackup',
                                                'DeviceProtection',
                                                'TechSupport',
                                                'StreamingTV',
                                                'StreamingMovies',
                                                'Contract',
                                                'PaperlessBilling',
                                                'PaymentMethod'
                                                ]
                        ):
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
    df_churn = df_churn.replace(
                                ' ', 
                                np.nan
                                ).dropna()

    prod_ids = df_churn['ProdID'].values.astype('int')
    df_churn = df_churn.drop(columns=['ProdID'])
    
    for col in categorical_features:
        df_churn[col] = df_churn[col].map(label_encoder_churn[col])

    df_churn = df_churn.astype('float')
    return df_churn, prod_ids

# Function to preprocess sentiment data
def preprocess_sentiment_data(
                            max_length = 60
                            ):
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
    df_sentiment = df_sentiment.replace(
                                        ' ', 
                                        np.nan
                                        ).dropna()

    prod_ids = df_sentiment['ProdID'].values.astype('int')
    df_sentiment = df_sentiment.drop(columns=['ProdID'])

    X = tokenizer_sentiment.texts_to_sequences(df_sentiment['Text'])
    X = tf.keras.preprocessing.sequence.pad_sequences(
                                                    X, 
                                                    maxlen=max_length
                                                    )
    return X, prod_ids

# Function to perform churn prediction on the preprocessed churn data
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
    df_churn, prod_ids_churn = preprocess_churn_data() 
    prediction = model_churn.predict(df_churn)
    
    df_inf_churn = pd.DataFrame(
                                {
                                    'ProdID': prod_ids_churn,
                                    'Churn': prediction
                                }
                                )
    churn_inference = db_sa['churn_inference']
    churn_inference.drop()
    churn_inference.insert_many(json.loads(df_inf_churn.to_json(orient='records')))

# Function to perform churn prediction on the preprocessed sentiment data
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
    df_inf_sentiment = pd.DataFrame(
                                    {
                                        'ProdID': prod_ids_sentiment,
                                        'Sentiment': prediction
                                    }
                                    )
    sentiment_inference = db_sa['sentiment_inference']
    sentiment_inference.drop()
    sentiment_inference.insert_many(json.loads(df_inf_sentiment.to_json(orient='records')))

# Function to calculate product score based on churn and sentiment
def inference_product_score(
                            prod_id,
                            alpha = 0.5
                            ):
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
        recommendation = recommendation_plans['LOW']
    elif product_score <= 7:
        recommendation = recommendation_plans['MEDIUM']
    else:
        recommendation = recommendation_plans['HIGH']
    recommendation["Score"] = product_score
    
    # Return analysis results
    return {
        'Sentiment Score': sentiment_score,
        'Churn Score': churn_score,
        'Recommendation': recommendation
    }

def main():
    # Create a navigation menu
    page = st.sidebar.selectbox("Select page", ["Main", "Custom Product Batch Analysis"])

    # Home page
    if page == "Main":
        st.title('SentiChurn - Hybrid Sentiment & Churn Analysis')
        st.write('This is a web application that performs sentiment and churn analysis on customer reviews and churn data.')
        st.write('The application uses machine learning models to predict sentiment and churn for each product and provides recommendations based on the analysis.')
        st.write('The application also provides a product score that combines the sentiment and churn predictions to give an overall assessment of the product.')

        st.header('Refresh MongoDB')
        st.write('Click the button below to refresh with the latest churn and sentiment predictions.')
        if st.button('Refresh MongoDB'):
            inference_churn()
            inference_sentiment()
            st.write('MongoDB has been refreshed with the latest churn and sentiment predictions.')

        st.header('Product Analysis')
        st.write('Enter the Product ID to get the analysis for that product.')
        prod_id = st.number_input('Product ID', min_value=1, max_value=101)

        if st.button('Analyze Product'):
            try:
                rec_output = inference_product_score(prod_id)
                df_rec_output = pd.DataFrame(rec_output.items(), columns=['Recommendation', 'Action'])
                st.dataframe(df_rec_output, hide_index=True)
            except:
                st.write('Product ID not found in the database. Please enter a valid Product ID.')

    elif page == "Custom Product Batch Analysis":
        st.title('Custom Product Analysis')
        
        # File uploader for custom product data
        uploaded_file = st.file_uploader("Upload CSV file for custom product analysis", type=["csv"])

        if uploaded_file is not None:
            try:
                # Read the uploaded CSV file
                custom_df = pd.read_csv(uploaded_file)
                
                # Initialize a list to store analysis results for each product
                analysis_results = []
                
                # Loop through each row of the DataFrame
                for index, row in custom_df.iterrows():
                    # Extract details of the custom product from the row
                    custom_data = {
                        "Text": row["Text"],
                        "gender": row["gender"],
                        "SeniorCitizen": row["SeniorCitizen"],
                        "Partner": row["Partner"],
                        "Dependents": row["Dependents"],
                        "tenure": row["tenure"],
                        "PhoneService": row["PhoneService"],
                        "MultipleLines": row["MultipleLines"],
                        "InternetService": row["InternetService"],
                        "OnlineSecurity": row["OnlineSecurity"],
                        "OnlineBackup": row["OnlineBackup"],
                        "DeviceProtection": row["DeviceProtection"],
                        "TechSupport": row["TechSupport"],
                        "StreamingTV": row["StreamingTV"],
                        "StreamingMovies": row["StreamingMovies"],
                        "Contract": row["Contract"],
                        "PaperlessBilling": row["PaperlessBilling"],
                        "PaymentMethod": row["PaymentMethod"],
                        "MonthlyCharges": row["MonthlyCharges"],
                        "TotalCharges": row["TotalCharges"]
                    }
                    print(custom_data)
                    # Call function to analyze custom product and append the analysis results
                    analysis_results.append(analyze_custom_product(custom_data))
                
                # Display the analysis results outside the loop
                for index, result in enumerate(analysis_results):
                    # Access prodid from the DataFrame
                    prodid = custom_df.iloc[index]["ProdID"]
                    st.write(f"Analysis for Product {prodid}:")
                    st.write(result)
    
            except Exception as e:
                st.error(f'Error: {e}')
if __name__ == '__main__':
    main()