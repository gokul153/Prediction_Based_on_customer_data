import streamlit  as st
import numpy as np

from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
print("Lets practice with stream lit")
import tensorflow as tf
print(tf.__version__)
## load trained model
model = load_model('model.h5')

## Load endoders and scalers

##load the encoder and scaler 
with open('label_encoder_gender.pk1','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pk1','rb') as file:
    label_encoder_geo = pickle.load(file)   

with open('scaler.pk1','rb') as file:
    scaler = pickle.load(file)

 ## streamlit app
st.title('Customer churn predictionn')

with st.form("customer_form"):
    CreditScore = st.number_input('Credit Score', value=619)
    Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'], index=0)
    Gender = st.selectbox('Gender', ['Female', 'Male'], index=0)
    Age = st.number_input('Age', value=42)
    Tenure = st.number_input('Tenure', value=2)
    Balance = st.number_input('Balance', value=0.0)
    NumOfProducts = st.number_input('Number of Products', value=1)
    HasCrCard = st.selectbox('Has Credit Card?', [1, 0], index=0)
    IsActiveMember = st.selectbox('Is Active Member?', [1, 0], index=0)
    EstimatedSalary = st.number_input('Estimated Salary', value=101348.88)
     # Submit button
    submitted = st.form_submit_button("Submit")

if submitted:
    # Store data in a dictionary
    print("Form Submitted")
    data = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }
    


    # Display the DataFrame
    st.write("Submitted Data:")
    st.write(data)

    ## one hot  
    encoder_geo = label_encoder_geo.transform([[data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(encoder_geo ,columns=label_encoder_geo.get_feature_names_out(['Geography']))
    st.write(geo_encoded_df)
    # encoder_geo = label_encoder_geo.transform([[df['Geography']]]).toarray()
    # geo_encoded_df = pd.DataFrame(encoder_geo ,columns=label_encoder_geo.get_feature_names_out(['Geography']))
    print(geo_encoded_df)
    df = pd.DataFrame([data])
    df['Gender'] = label_encoder_gender.transform(df['Gender'])
    print(df)
    df_concated_geo = pd.concat([df.drop("Geography",axis=1),geo_encoded_df],axis=1)
    print("concatented geo {}",df_concated_geo)
    ## scalling the input data 
    input_scaled = scaler.transform(df_concated_geo)
    input_scaled
    ## predict the churn
    prediction = model.predict(input_scaled)
    prediction_prob= prediction[0][0]
    st.write("Prediction probability {}",prediction_prob)
    if prediction_prob > 0.5:
        st.write("The customer is likely to leave the bank")
    else:
        st.write("The customer is not likely to move out of bank")

    