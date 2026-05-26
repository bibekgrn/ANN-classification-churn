import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle

#load the trained model
model=tf.keras.models.load_model('model.h5')


##load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title('custoomer churn prediction')

#user input
geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])#0 is like pre selected in selectbox
#geography-----topic
#one_hot_encoder_geo.categories_[0]-----france germany spain options to choose in checkbox
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
Balance=st.number_input('balance')
creditscore=st.number_input('creditscore')
estimatedsalary=st.number_input('estimatedsalary')
tenure=st.slider('tenure',0,10)
numofproducts=st.slider('number of products',1,4)
hascrcard=st.selectbox('has credit card',[0,1])
isactivemember=st.selectbox('isactive member',[0,1])

##input the data
input_data=pd.DataFrame({
    'CreditScore':[creditscore],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[Balance],
    'NumOfProducts':[numofproducts],
    'HasCrCard':[hascrcard],
    'IsActiveMember':[isactivemember],
    'EstimatedSalary':[estimatedsalary]
})

#onehotencode 'Georgraphy'
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
)


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)#axis=1 join side by side not top
#[input_data.reset_index(drop=True),geo_encoded_df]=[inputdata,geoencodeddf]---resetindex avoids index mismatch


#scale the input data
input_data_scaled=scaler.transform(input_data)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'churn probablility:{prediction_prob:.2f}')

if prediction_prob>0.5:
    st.write('the customer is likely to churn')
else:
    st.write('the customer is not likely to churn')
