#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:56:09 2024

@author: priyapatil
"""

# Importing necessary libraries
import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import time

# Loading the trained model
with open("lr_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Function to preprocess the input data
def pre_process(df):
    # Set Proper Precision : Round float columns to 2 decimal places
    cols_float = ['MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration', 'UserRating']
    df[cols_float] = df[cols_float].round(2)

    # Label Encoding : Mapping categorical values to numerical values
    cols_to_encode = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType', 'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled']
    
    for col in cols_to_encode:
        if col == 'SubscriptionType':
            df['SubscriptionType'] = df['SubscriptionType'].map({'Basic': 0, 'Premium': 1, 'Standard': 2})
                
        if col == 'PaymentMethod':
            df['PaymentMethod'] = df['PaymentMethod'].map({'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3})
                
        if col == 'PaperlessBilling':
            df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
                
        if col == 'ContentType':
            df['ContentType'] = df['ContentType'].map({'Both': 0, 'Movies': 1, 'TV Shows': 2})
        
        if col == 'MultiDeviceAccess':
            df['MultiDeviceAccess'] = df['MultiDeviceAccess'].map({'No': 0, 'Yes': 1})
                
        if col == 'DeviceRegistered':
            df['DeviceRegistered'] = df['DeviceRegistered'].map({'Computer': 0, 'Mobile': 1, 'TV': 2, 'Tablet': 3})
                
        if col == 'GenrePreference':
            df['GenrePreference'] = df['GenrePreference'].map({'Action': 0, 'Comedy': 1, 'Drama': 2, 'Fantasy': 3, 'Sci-Fi': 4})
                
        if col == 'Gender':
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
           
        if col == 'ParentalControl':
            df['ParentalControl'] = df['ParentalControl'].map({'No': 0, 'Yes': 1})
        
        if col == 'SubtitlesEnabled':
            df['SubtitlesEnabled'] = df['SubtitlesEnabled'].map({'No': 0, 'Yes': 1})
  
    return df

# Function to plot feature importance as bar chart
def plot_feature_importance_bar(feature_importance):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    st.pyplot(plt)

# Function to plot feature importance as pie chart
def plot_feature_importance_pie(feature_importance):
    plt.figure(figsize=(8, 8))
    plt.pie(feature_importance['Importance'], labels=feature_importance['Feature'], autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
    plt.title('Feature Importance')
    st.pyplot(plt)

# Function to show prediction result and analysis plot
def show_prediction_result(final_df):
    selected_features = ['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration', 'ContentDownloadsPerMonth', 'GenrePreference', 'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize']
    st.write("") 
    with st.spinner('Predicting...'):
        pred =  loaded_model.predict(final_df) 
        time.sleep(2)  
        
    # Display the result in form of string
    if pred == 0:
        st.success("This customer is unlikely to churn!")
    else:
        st.error("This customer is likely to churn!")


    feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': loaded_model.coef_[0]})
    feature_importance['Importance'] = feature_importance['Importance'].abs()  # Taking absolute values

    # Ploting the bar chart and pie chart
    st.subheader('Impact of each feature affecting the churn decision.')
    plot_feature_importance_bar(feature_importance)
    plot_feature_importance_pie(feature_importance)


# Main function to run the Streamlit web application
def main():
    # App title
    st.title('Customer Churn Predictor')

    # Input fields for Customer Details
    st.subheader('Customer Details') 
    customer_id = st.text_input('Customer ID', '')
    gender = st.radio('Gender', ['Male', 'Female'])

    # Input fields for User Preferences
    st.subheader('User Preferences')
    device_registered = st.selectbox('Device Registered', ['Mobile','Tablet','TV','Computer'])
    md_access = st.radio('Multi-Device Access', ['Yes','No'])
    content_type = st.selectbox('Content Type', ['TV Shows','Movies','Both']) 
    genre = st.selectbox('Genre', ['Sci-Fi','Drama','Action','Comedy','Fantasy'])
    parental_control = st.radio('Parental Control', ['Yes','No'])
    subtitles = st.radio('Enable Subtitles', ['Yes','No'])

    # Input fields for Billing Information
    st.subheader('Billing Information')
    subscription = st.radio('Subscription Type', ['Premium','Standard','Basic'])
    payment = st.selectbox('Payment Method', ['Mailed check', 'Credit card', 'Electronic check', 'Bank transfer'])
    paperless = st.radio('Parperless-Billing', ['Yes','No'])
    
    # Input field for User Rating
    st.subheader('User Rating') 
    user_rating = st.slider('User Rating (1 - 5)', 1.0, 5.0, step=0.01, format="%.2f")

    # Button to trigger prediction
    if st.button('Predict'):
        #with st.spinner('Predicting...'):
            accountage = random.randint(1,119)
            monthly_charges = random.uniform(4.99, 19.98)
            total_charges = random.uniform(4.99, 2378.72)   
            viewing_hrs_perweek = random.uniform(1,39.99)
            average_viewing = random.uniform(5,179.99) 
            content_download = random.randint(0,49)  
            support_tickets = random.randint(0,9) 
            watchlist_size = random.randint(0,24) 
           
            data = {
                'AccountAge' : [accountage],
                'MonthlyCharges' : [monthly_charges],
                'TotalCharges':[total_charges],
                'SubscriptionType': [subscription],
                'PaymentMethod': [payment],
                'PaperlessBilling': [paperless],
                'ContentType': [content_type],
                'MultiDeviceAccess': [md_access],
                'DeviceRegistered': [device_registered],
                'ViewingHoursPerWeek':[viewing_hrs_perweek],
                'AverageViewingDuration':[average_viewing],
                'ContentDownloadsPerMonth':[content_download],
                'GenrePreference': [genre],
                'UserRating': [user_rating],
                'SupportTicketsPerMonth':[support_tickets],
                'Gender': [gender],
                'WatchlistSize':[watchlist_size],
                'ParentalControl': [parental_control],
                'SubtitlesEnabled': [subtitles]
            }

            df = pd.DataFrame(data)
            processed_df = pre_process(df)
            selected_features = ['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration', 'ContentDownloadsPerMonth', 'GenrePreference', 'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize']
            final_df = processed_df[selected_features]
            
            show_prediction_result(final_df)
if __name__ == "__main__":
    main()
