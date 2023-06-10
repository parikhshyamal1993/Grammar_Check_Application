import streamlit as st
from transformers import pipeline
import numpy as np

Lable_Map = {"LABEL_0":"Grammatical mistake","LABEL_1":"Sentence is fine"}

classifier = pipeline("text-classification",r"C:\Users\Shyamal Parikh\Downloads\model_save-20230605T134409Z-001\model_save")


st.write(
    """
    # AI Grammer Check

    Check your sentance here !
    """
)

query = st.text_input("Senence","")

if query != "" :
    output = classifier(query)
    response = Lable_Map[str(output[0]['label'])]
    print("response:",response)
    st.write(f"{response}")