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
from langchain.document_loaders import DirectoryLoader, TextLoader


def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    # create docs file from all companies
    loader = DirectoryLoader('./data/', glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    # add metadata for company and industry
    industry = ['IT', 'IT']
    company = ['Amazon', 'Apple']
    for i, doc in enumerate(docs):
        doc.metadata['industry'] = industry[i]
        doc.metadata['company'] = company[i]

    # prepare chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    len(splits)

    topsummarizer = TopicSummarizer(api_key)
    topsummarizer.docs = splits
    print(len(splits))
    # --- if not using embedding and only for one company
    _company = [x for x in splits if x.metadata['company'] == 'Amazon']
    topsummarizer.docs = _company
    topsummarizer.simple_question(PRED_PROMPT_CONSUEMR_SPENDING, topic='consumer spending')
    topic_found = topsummarizer.extract_topics(TOPIC_PROMPT_MAP_TEMPLATE,
                                               TOPIC_PROMPT_COMBINE_TEMPLATE,
                                               structured=False,
                                               schema=TOPIC_SCHEMA)
    topics= pd.DataFrame(topic_found)
    topics.to_csv('amazon.csv')
    # --- The end

    # --- If using vector db
    # Create Vector db first time or load if exists
    # topsummarizer.create_vectordb()
    #topsummarizer.vectordb = Chroma.load('docs/chroma/appl/')
    # prompt = 'find a consumer spending related information, like consumer confidence, affordability'
    # summary = topsummarizer.retrieve_information(topic='Consumer Spending',
    #                                              retrievefirst=True,
    #                                              fetch_topic=prompt)

    # topsummarizer.retrieve_information_template(SUMMARY_TEMPLATE,
    #                                       topics=['consumer spending and confidence', 'labor cost'],
    #                                       verbose=True)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
