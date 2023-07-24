import streamlit as st
import pandas as pd
from chatcompletion import *
import os
import openai
from dotenv import load_dotenv
import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
from utils import write_list_to_file
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
from prompts import (
    TOPIC_PROMPT_MAP_TEMPLATE,
    TOPIC_PROMPT_COMBINE_TEMPLATE,
    TOPIC_PROMPT_COMBINE_TEMPLATE2,
    TOPIC_SCHEMA,
    SUMMARY_TEMPLATE,
    PRED_PROMPT_CONSUEMR_SPENDING,
    PRED_PROMPT_LABOR_COST)


load_dotenv()
openai.api_key =os.getenv('OPENAI_API_KEY')
testsummary = Summarizer()


# Check if path is not empty
st.header("Earnings Transcript Summarization:books:")

@st.cache_data
def load_data(data_path):
    docs = testsummary.prepare_data_from_files(data_path)
    return docs

# Sidebar for data upload
st.sidebar.header("Load Data")
data_path = st.sidebar.text_input("Enter the path to your data file")

# Sidebar for summarizaton argument
st.sidebar.header("Summarization")
summary_method = st.sidebar.selectbox("Choose a method", ['Auto', 'Topic'])
if summary_method == 'Topic':
    topic = st.sidebar.text_input("Enter the topic")


if data_path:
    try:
        docs = load_data(data_path)
        print('load raw data {}'.format(len(docs)))
        st.sidebar.success("Data loaded successfully")

        # Give option to subset the data
        st.sidebar.header("Subset Data")
        company_list = list(set([x.metadata['company'] for x in docs]))
        col_to_filter = st.sidebar.selectbox("Choose a company", company_list)
        print(col_to_filter)
        if st.sidebar.button('summary'):
            subdocs = [x for x in docs if col_to_filter in x.metadata['company']]
            if 'subdocs' in locals():
                st.write(subdocs[0])
                if summary_method == 'Auto':
                    st.write('Generating an auto summary:')
                    _,_,summary = testsummary.doc_summary(TOPIC_PROMPT_MAP_TEMPLATE, TOPIC_PROMPT_COMBINE_TEMPLATE2,
                                                       mydocs=subdocs)
                    st.write("Results:", summary)
                if summary_method == 'Topic':
                    st.write("Generating {} summary".format(topic))
                    _, output = testsummary.topic_extract_summary(PRED_PROMPT_CONSUEMR_SPENDING, topic, mydocs=subdocs)
                    st.write("topic_summary:", output)

    except FileNotFoundError:
        st.sidebar.error("File not found. Please check the path and try again")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Main area
#
# st.header("Earnings Transcript Summarization")
# if st.button('Write a summary'):
# #     a,b,c = testsummary.doc_summary(TOPIC_PROMPT_MAP_TEMPLATE, TOPIC_PROMPT_COMBINE_TEMPLATE2, mydocs=mydocs)
