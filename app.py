import streamlit as st
import pandas as pd
from inference import inference_churn, inference_sentiment, inference_product_score, analyze_custom_product

def main():
    # Create a navigation menu
    page = st.sidebar.selectbox("Select page", ["Main", "Custom Product Analysis"])

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

    elif page == "Custom Product Analysis":
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
                    rec_output = analyze_custom_product(custom_data)
                    rec_output["Product ID"] = custom_df.iloc[index]["ProdID"]
                    df_rec_output = pd.DataFrame(rec_output.items(), columns=['Recommendation', 'Action'])
                    st.dataframe(df_rec_output, hide_index=True)
    
            except Exception as e:
                st.error(f'Error: {e}')
if __name__ == '__main__':
    main()
