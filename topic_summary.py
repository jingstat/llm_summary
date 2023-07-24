from dotenv import load_dotenv
from langchain.chains import RetrievalQA, create_extraction_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from utils import write_list_to_file
class TopicSummarizer:
    def __init__(self, api_key, ric=None):
        self.api_key = api_key
        # Initialize models here, replace 'YourModel' with the actual model classes
        self.llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=self.api_key,
                  model_name="gpt-3.5-turbo-0613",
                  request_timeout = 180
                )

        self.llm4 = ChatOpenAI(temperature=0,
                  openai_api_key=self.api_key,
                  model_name="gpt-4-0613",
                  request_timeout = 180
                 )
        # Add any necessary constants here
        self.ric = ric
        self.docs = None
        self.vectordb = None


    def prepare_data(self, data, chunk_size=1500, chunk_overlap=150):
        # Load and prepare the data for topic extraction
        full_text = '\n\n'.join(data)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.docs = text_splitter.create_documents([full_text], [{'1': 1}])
        print(f"You have {len(self.docs)} docs")
        return self.docs

    def extract_topics(self, topic_prompt_map_template, topic_prompt_combine_template, structured=False, schema=None):
        # Run the extraction and summarization chains
        # build chat_prompt_map
        llm = self.llm3
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(topic_prompt_map_template)
        human_template = "Transcript: {text}"  # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt_map = ChatPromptTemplate.from_messages(
            messages=[system_message_prompt_map, human_message_prompt_map])

        # build chet_prompt_combine
        system_message_prompt_combine = SystemMessagePromptTemplate.from_template(topic_prompt_combine_template)
        human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(
            messages=[system_message_prompt_combine, human_message_prompt_combine])

        chain = load_summarize_chain(llm,
                                     chain_type="map_reduce",
                                     map_prompt=chat_prompt_map,
                                     combine_prompt=chat_prompt_combine,
                                     )
        topics_found = chain.run({"input_documents": self.docs})
        if structured and topics_found:
            s_chain = create_extraction_chain(schema, llm)
            topics_found = s_chain.run(topics_found)
        return topics_found


    def create_vectordb(self):
        openai_embed = OpenAIEmbeddings(openai_api_key=self.api_key)
        persist_directory = 'docs/chroma/'
        if os.path.exists(persist_directory):
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=openai_embed)
        else:
        #!rm - rf. / docs / chroma  # remove old database files if any
            vectordb = Chroma.from_documents(
                documents=self.docs,
                embedding=openai_embed,
                persist_directory=persist_directory
            )
            self.vectordb= vectordb
        return True

    def retrieve_information(self, topic, retrievefirst=False, fetch_topic=None):
        # Retrieve and display information for specific topics
        question = "please summarize {} related information".format(topic)
        if retrievefirst:
            subset = self.vectordb.similarity_search(fetch_topic,
                                                     k=3,
                                                     filter={'company': 'Apple'})

            chain = load_qa_chain(self.llm3, chain_type="stuff")
            output = chain.run(input_documents=subset, question=question)
        else:
            qa_chain = RetrievalQA.from_chain_type(self.llm3,
                                                   retriever=self.vectordb.as_retriever(),
                                                   )
            output = qa_chain({"query": question})
        return output

    def retrieve_information_template(self, summary_template, topics, verbose=False):
        messages = [
            SystemMessagePromptTemplate.from_template(summary_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        # This will pull the two messages together and get them ready to be sent to the LLM through the retriever
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
        qa = RetrievalQA.from_chain_type(llm=self.llm3,
                                         chain_type="stuff",
                                         retriever=self.vectordb.as_retriever(),
                                         chain_type_kwargs={
                                             'verbose': verbose,
                                             'prompt': CHAT_PROMPT
                                         })
        for topic in topics:
            question = "please summarize {} related information".format(topic)
            expanded_topic = qa.run(question)
            print(expanded_topic)
            print("\n\n")
        return True

    def simple_question(self, pred_prompt, topic=None):
        #
        pred_prompt_template = PromptTemplate(template=pred_prompt, input_variables=["text"])
        pred_llm_chain = LLMChain(llm=self.llm3, prompt=pred_prompt_template)
        output_list = [pred_llm_chain.run(x.page_content) for x in self.docs]
        select = [True if 'yes' in x.lower() else False for x in output_list]
        subset = [item for item, mask in zip(self.docs, select) if mask]
        print(len(subset))
        print([x for x in subset])
        file_name = 'output.txt'
        write_list_to_file(subset, file_name)
        #print(subset)
        if topic:
            if len(subset)>0:
                question = "please write a summary (5 sentences or less) about {}. Do not respond with information that isn't relevant to the topic that the user gives you".format(topic)
                #question = "please summarize {} related information, use bulline points".format(topic)
                chain = load_qa_chain(self.llm3, chain_type="stuff")
                output = chain.run(input_documents=subset, question=question)
                write_list_to_file([output], 'amazon_summary.txt')
            else:
                output = 'This is no {} related information'.format(topic)
            print(output)
        return True
