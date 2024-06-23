import streamlit as st
import pickle
import numpy as np

# Load the trained model and dataframe
RF = pickle.load(open('RF_final_23-06-2024.pkl', 'rb'))
df1 = pickle.load(open('df1_23-06-2024.pkl', 'rb'))

# Title and header
st.title('Car Price Prediction')
st.header('Fill The Used Car Details to Predict The Price')

# User inputs
Car_Name = st.selectbox('Car_Name', df1['Car_Name'].unique())
Model = st.selectbox('Model', df1['Model'].unique())
Type = st.selectbox('Type', df1['Type'].unique())
year = st.selectbox('Year', df1['year'].unique())
km_driven = st.number_input('KM(between 1.0 - 806599.0)')
fuel = st.selectbox('Fuel', df1['fuel'].unique())
seller_type = st.selectbox('Seller Type', df1['seller_type'].unique())
transmission = st.selectbox('Transmission', df1['transmission'].unique())
owner = st.selectbox('Owner', df1['owner'].unique())

# Predict button
if st.button('Predict'):
    try:
        # Convert categorical variables to numerical
        car_name_map = {name: i for i, name in enumerate(df1['Car_Name'].unique())}
        model_map = {name: i for i, name in enumerate(df1['Model'].unique())}
        type_map = {name: i for i, name in enumerate(df1['Type'].unique())}
        fuel_map = {name: i for i, name in enumerate(df1['fuel'].unique())}
        seller_type_map = {name: i for i, name in enumerate(df1['seller_type'].unique())}
        transmission_map = {name: i for i, name in enumerate(df1['transmission'].unique())}
        owner_map = {name: i for i, name in enumerate(df1['owner'].unique())}

        Car_Name = car_name_map.get(Car_Name, 'Enter Correct Car_Name')
        Model = model_map.get(Model, 'Enter Correct Model')
        Type = type_map.get(Type, 'Enter Correct Type')
        fuel = fuel_map.get(fuel, 'Select Valid Fuel Type')
        seller_type = seller_type_map.get(seller_type, 'Select Valid Category')
        transmission = transmission_map.get(transmission, 'Select Valid Type')
        owner = owner_map.get(owner, 'Select Valid Category')

        # Check if all inputs are valid
        if all(isinstance(val, int) for val in [Car_Name, Model, Type, fuel, seller_type, transmission, owner]):
            # Make prediction
            input_data = np.array([[Car_Name, Model, Type, year, km_driven, fuel, seller_type, transmission, owner]])
            prediction = RF.predict(input_data)

            # Display prediction
            prediction_text = f"Predicted Price: {prediction[0]}"
            st.write(prediction_text)
        else:
            st.write("Please enter valid values for all input fields.")
    except Exception as e:
        st.write("An error occurred:", e)