import pandas as pd
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np

age_options={
    1:'Young(1)',
    2:'Adult(2)',
    3:'Old(3)'
}

race_options={
    1:'White(1)',
    2:'Black(2)',
    3:'Asian or Pacific Islander(3)',
    4:'American Indian/Alaska Native(4)',
    5:'unknown(5)'        
}

marital_options={
    1:'Single(1)',
    2:'Married(2)',
    3:'Divorced(3)',
    4:'Widowed(4)',
    5:'other(5)'
}

site_options={
    1:'Head of pancreas(1)',
    2:'Body of pancreas(2)',
    3:'Tail of pancreas(3)',
    4:'other(4)'
}

T_options={ 
 1:'T0-T2(1)',
 2:'T3-T4(2)',
 3:'unknown(3)'
 }

N_options={ 
 1:'N0(1)',
 2:'N1~N2(2)',
 3:'unknown(3)'
 }

grade_options={
    1:'Grade I(1)',
    2:'Grade II(2)',
    3:'Grade III(3)',
    4:'Grade IV(4)',
    5:'unknown(5)'
    }

histological_options={
    1:'Adenocarcinoma(1)',
    2:'Infiltrating duct carcinoma(2)',
    3:'Neuroendocrine carcinoma(3)',
    4:'Carcinoid tumor(4)',
    5:'other(5)'
    }


multifocality_options={
    1:'Solitary tumor(1)',
    2:'Multifocal tumor(2)',
    }

feature_names = [ "age", "race", "sex", "marital","site", "T",  "N","grade","histological","multifocality"]

st.header(" Distant metastasis of pancreatic cancer predictor app")

age=st.selectbox("age:", options=list(age_options.keys()), format_func=lambda x: age_options[x])
race=st.selectbox("race:", options=list(race_options.keys()), format_func=lambda x: race_options[x])
sex = st.selectbox("sex ( 1=Male, 2=Female):", options=[1, 2], format_func=lambda x: 'Female (2)' if x == 2 else 'Male (1)')
site=st.selectbox("site:", options=list(site_options.keys()), format_func=lambda x: site_options[x])
marital=st.selectbox("marital:", options=list(marital_options.keys()), format_func=lambda x: marital_options[x])
T=st.selectbox("T:", options=list(T_options.keys()), format_func=lambda x: T_options[x])
N=st.selectbox("N:", options=list(N_options.keys()), format_func=lambda x: N_options[x])
grade=st.selectbox("grade:", options=list(grade_options.keys()), format_func=lambda x: grade_options[x])
histological=st.selectbox("histological:", options=list(histological_options.keys()), format_func=lambda x: histological_options[x])
multifocality=st.selectbox("multifocality:", options=list(multifocality_options.keys()), format_func=lambda x: multifocality_options[x])


feature_values = [age, race, sex, site, marital,T, N,grade,histological,multifocality]   

features = np.array([feature_values])

if st.button("Submit"):
    clf = open("xgboost.pkl","rb")
    s=pickle.load(clf)
    predicted_class = s.predict(features)[0]
    predicted_proba = s.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        st.write('According to our model, you have a high risk of distant metastasis')
    else:
        st.write('According to our model, you have a low risk of distant metastasis')  

    explainer = shap.Explainer(s)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

  
