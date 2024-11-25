import streamlit as st
import pandas as pd
import joblib

st.title('Formulário hospital utilizar e indicar se o paciente tem ou não insuficiência cardíaca')

age = st.text_input('Idade', '')
sex = st.selectbox('Sexo', ['Masculino', 'Feminino'])
anaemia = st.selectbox('Anemia?', ['Sim', 'Não'])
diabetes = st.selectbox('Diabetico?', ['Sim', 'Não'])
high_blood_pressure = st.selectbox('Hipertenso?', ['Sim', 'Não'])
smoking = st.selectbox('Fuma?', ['Sim', 'Não'])

creatinine_phosphokinase = st.text_input('Creatinina Fosfoquinase mcg/L (min: 32, max: 294)', '')
ejection_fraction = st.text_input('Fração de Ejeção (Porcentagem) (min: 50, max: 70)', '')
platelets = st.text_input('Plaquetas kiloplatelets/mL (min: 150000, max: 450000)', '')
serum_creatinine = st.text_input('Creatinina Sérica mg/dL (min: 0.6, max: 1.2)', '')
serum_sodium = st.text_input('Sódio Sérico mEq/L (min: 135, max: 145)', '')


all_filled = age and anaemia and creatinine_phosphokinase and diabetes and ejection_fraction and high_blood_pressure and platelets and serum_creatinine and serum_sodium and sex and smoking


if st.button('Exemplo'):
    example = pd.read_csv('example.csv')
    st.dataframe(example)
    model = joblib.load('pipeline_model.joblib')
    prediction = model.predict(example)[0]
    if prediction == 1:
        st.success('O paciente tem alta probabilidade insuficiência cardíaca')
        st.stop()
    else:
        st.success('O paciente nao tem alta probabilidade de insuficiência cardíaca')
        st.stop()

# Botão para submeter os dados
if st.button('Submeter') and all_filled:
    sex_input = 1 if sex == 'Masculino' else 0
    diabetes_input = 1 if diabetes == 'Sim' else 0
    smoking_input= 1 if smoking == 'Sim' else 0
    high_blood_pressure_input = 1 if high_blood_pressure == 'Sim' else 0
    anaemia_input = 1 if anaemia == 'Sim' else 0

    data = pd.DataFrame([
        {
            'age': int(age),
            'anaemia': int(anaemia_input),
            'creatinine_phosphokinase': int(creatinine_phosphokinase),
            'diabetes': diabetes_input,
            'ejection_fraction': int(ejection_fraction),
            'high_blood_pressure': int(high_blood_pressure_input),
            'platelets': int(platelets),
            'serum_creatinine': float(serum_creatinine),
            'serum_sodium': int(serum_sodium),
            'sex': sex_input,
            'smoking': smoking_input
        },
    ]
    )
    #st.dataframe(data)
    model = joblib.load('pipeline_model.joblib')

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success('O paciente tem alta probabilidade insuficiência cardíaca')
        st.stop()
    else:
        st.success('O paciente nao tem alta probabilidade de insuficiência cardíaca')
        st.stop()
else:
    st.error('Por favor, preencha todos os campos.')
    st.stop()