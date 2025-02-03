import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = "LinearRegressionModel.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the dataset to fetch unique names and companies
df = pd.read_csv("cleaned_car.csv")
unique_companies = df["company"].unique().tolist()

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter car details to predict its price.")

# Company selection
company = st.selectbox("Company", unique_companies)
filtered_names = df[df["company"] == company]["name"].unique().tolist()

# Car Name selection (filtered based on company)
name = st.selectbox("Car Name", filtered_names)

# Other input fields
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, value=2015)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])

# Convert categorical input to numerical (example encoding)
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3}
fuel_type_encoded = fuel_mapping[fuel_type]

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                              columns=["name", "company", "year", "kms_driven", "fuel_type"])
    prediction = model.predict(input_data)
    predicted_price = max(prediction[0], 0)  # Ensure price is not negative
    st.success(f"Estimated Car Price for {name}: ₹{predicted_price:,.2f}")
    
    # Show similar listings
    similar_cars = df[(df["company"] == company) & (df["year"] >= year - 2) & (df["year"] <= year + 2)]
    if not similar_cars.empty:
        st.write("### Similar Cars for Reference:")
        st.dataframe(similar_cars[["name", "company", "year", "kms_driven", "fuel_type"]])
