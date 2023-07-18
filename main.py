from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA, create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from prompts import (
    TOPIC_PROMPT_MAP_TEMPLATE,
    TOPIC_PROMPT_COMBINE_TEMPLATE,
    TOPIC_SCHEMA,
    SUMMARY_TEMPLATE,
    PRED_PROMPT_CONSUEMR_SPENDING,
    PRED_PROMPT_LABOR_COST)
from topic_summary import TopicSummarizer
from dotenv import load_dotenv
from utils import load_text, create_sentences, create_chunks
def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    topsummarizer = TopicSummarizer(api_key, 'appl')
    txt_filepath = 'data/amzn_earningtrans.txt' #'data/AAPL_Q2_2023.txt' #'amzn_earningtrans.txt'
    segments = load_text(txt_filepath)
    sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=100)
    chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
    chunks_text = [chunk['text'] for chunk in chunks]
    print(len(chunks_text))
    docs = topsummarizer.prepare_data(data=chunks_text, chunk_size=1500, chunk_overlap=150)
    print(docs[0])

    # topic_found = topsummarizer.extract_topics(TOPIC_PROMPT_MAP_TEMPLATE,
    #                                            TOPIC_PROMPT_COMBINE_TEMPLATE,
    #                                            structured=True,
    #                                            schema=TOPIC_SCHEMA)
    #print(topic_found)
    # Create Vector db first time or load if exists
    topsummarizer.create_vectordb()
    #topsummarizer.vectordb = Chroma.load('docs/chroma/appl/')
    # prompt = 'find a consumer spending related information, like consumer confidence, affordability'
    # summary = topsummarizer.retrieve_information(topic='Consumer Spending',
    #                                              retrievefirst=True,
    #                                              fetch_topic=prompt)

    # topsummarizer.retrieve_information_template(SUMMARY_TEMPLATE,
    #                                       topics=['consumer spending and confidence', 'labor cost'],
    #                                       verbose=True)

    topsummarizer.simple_question(PRED_PROMPT_LABOR_COST, topic='labor cost')

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
