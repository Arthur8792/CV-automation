"""
Code for streamlit app.
"""
import streamlit as st
import time
from main import main

# Add session variables
if 'ready_download' not in st.session_state:
    st.session_state['ready_download'] = None 

# Title and app presentation
st.title("Créateur de CV Silamir")
st.write("Cette application permet de convertir un CV source PDF en un CV Silamir Word.")
st.write("NOTE : la conversion peut prendre de 2 à 10mn selon la taille du CV.")

# Create file uploader and converter button
uploaded_file = st.file_uploader("Sélectionner le CV", accept_multiple_files=False, type=['pdf'])

if uploaded_file is not None:
    converter_button = st.button("Convertir", type='primary')
    if converter_button:

        # Create tmp file which contains the pdf CV
        cv_file = 'data/cv/cv.pdf'
        with open(cv_file,"+bw") as file:
            file.write(uploaded_file.getvalue())

        # Run pipeline
        with st.spinner('Conversion en cours...'):
            time.sleep(5)
            st.session_state["ready_download"] = main(cv_file)
        st.success("Conversion réalisée avec succès, vous pouvez télécharger le CV.")

    # Get Silamir CV in .docx
    if st.session_state["ready_download"] is not None:
        with open(st.session_state["ready_download"], "rb") as file:
            download_button = st.download_button(
                label = "Télécharger CV Silamir",
                data=file,
                file_name = "cv_silamir.docx",
                mime = "application/pdf",                
            )

