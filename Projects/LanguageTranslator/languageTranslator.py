# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:25:58 2022

@author: rupak
"""

#The googletrans API created by Google developers was used earlier to translate any text using Python. But now the new API from the Google developers known as google_trans_new is used for the same task.

from google_trans_new import google_translator
import streamlit as st
translator = google_translator()
st.title("Language Translator")
text = st.text_input("Enter a text")
translate = translator.translate(text, lang_tgt='fr')
st.write(translate)