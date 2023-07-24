import os
import openai

# --------------------------------------------------------------
# Ask ChatGPT a Question
# --------------------------------------------------------------

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

messages = [{'role': "system",
             'content': "you are a helpful assistant "
             },
            {
                "role": "user",
                "content": "When's the next flight from Amsterdam to New York?",
            },
            ]
# --------------------------------------------------------------
# Ask ChatGPT a Question
# --------------------------------------------------------------
def openai_completions(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages= [{'role':'user',
                    'content': prompt}],
    )
    output = completion.choices[0].message.content
    return output

def openai_completions_roles(sys_prompt,
                             human_prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages= [sys_prompt, human_prompt],
        temperature = 0
    )
    output = completion.choices[0].message.content
    return output

# --------------------------------------------------------------
# Summarizer class
# --------------------------------------------------------------


class Summarizer:
    def __init__(self):
        # Initialize models here, replace 'YourModel' with the actual model classes
        self.docs = None
        self.ric = 'other'


    def prepare_data_from_text(self, data, chunk_size=1000, chunk_overlap=0):
        """
        :param data: raw text
        :param chunk_size: 1000
        :param chunk_overlap: 0
        :return: self.docs with metadata
        """
        # Load and prepare the data for topic extraction
        full_text = '\n\n'.join(data)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.create_documents([full_text], [{'1': 1}])
        print(f"You have {len(self.docs)} docs")
        return self.docs

    def prepare_data_from_csv_paragraph(self, filepath):
        # Load and prepare the data for topic extraction
        """
        :param filepath: path to a list of txt files
        :return: self.docs with metadata
        """
        loader = CSVLoader(filepath)
        docs = loader.load()
        raw_data = pd.read_csv(filepath)
        for i, doc in enumerate(docs):
            _ric = raw_data.iloc[i]['ric'].replace('.','_')
            doc.metadata['ric']=_ric
            doc.metadata['paragraph_id']=raw_data.iloc[i]['paragraph_id']
        print(len(self.docs))
        return self.docs
    def prepare_data_from_files(self, filepath, chunk_size=1500, chunk_overlap=200):
        # Load and prepare the data for topic extraction
        """
        :param filepath: path to a list of txt files
        :param chunk_size:  1000
        :param chunk_overlap: 0
        :return: self.docs with metadata
        """
        loader = DirectoryLoader(filepath, glob='*.txt', loader_cls=TextLoader)
        docs = loader.load()
        for i, doc in enumerate(docs):
            _company = doc.metadata['source'].split('/')[-1]
            doc.metadata['company'] = _company
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(docs)
        print(len(self.docs))
        return self.docs

    def topic_extract_summary(self, pred_prompt, topic='Consumer Spending', mydocs=None, intermediate=True):
        """
        :param pred_prompt: To predict if paragraph is related to a topic
        :param topic: e.g 'consumer spending'
        :return:  related source, summary
        """
        if mydocs==None:
            mydocs = self.docs
        # Ask question for each chunk and extract related chunks
        output_list =[]
        print('extraction started')
        for x in mydocs:
            _output = openai_completions(prompt = pred_prompt.replace('{text}',x.page_content))
            output_list.append(_output)
            time.sleep(3)
#       output_list = [openai_completions(prompt = pred_prompt.replace('{text}',x.page_content)) for x in mydocs]
        select = [True if 'yes' in x.lower() else False for x in output_list]
        subset = [item for item, mask in zip(mydocs, select) if mask]
        print('extraction completed {} chunk'.format(len(subset)))
        if intermediate:
            print(len(subset))
            file_name = './output/{}_extracted.txt'.format(self.ric)
            write_list_to_file(subset, file_name)
        if topic:
            if len(subset)>0:
                extraction = ('\n\n').join([x.page_content for x in subset])
                summary_prompt = """please write a summary (5 sentences or less) about {} based on the Text. 
                Do not respond with information that isn't relevant to the topic that the user gives you.
                Text:[{}]""".format(topic, extraction)
                output = openai_completions(prompt=summary_prompt)
                summary_file_name =  './output/{}_extract_summary.txt'.format(self.ric)
                write_list_to_file([output], summary_file_name)
            else:
                output = 'There is no {} related information'.format(topic)
            print(output)
        print('final summarization completed')
        return subset, output

    def doc_summary(self, topic_prompt_map_template, topic_prompt_combine_template, mydocs=None):
        """

        :param topic_prompt_map_template:
        :param topic_prompt_combine_template:
        :param mydocs:
        :return: chunk based topic&summary, chunk source, final summary
        """
        # Run the extraction and summarization chains
        print('auto-sum-start...')
        if mydocs == None:
            mydocs= self.docs
        human_template = "Transcript: {text}"  # Simply just pass the text as a human message
        auto_topic_list =[]
        for x in mydocs:
            _output = openai_completions_roles(sys_prompt={'role': 'system', 'content': topic_prompt_map_template},
                                                    human_prompt={'role':'user',
                                                                  'content':human_template.replace('{text}', x.page_content)})
            auto_topic_list.append(_output)
            time.sleep(2)
        select = [False if 'no topics' in x.lower() else True for x in auto_topic_list]
        auto_topic_subset = [x for x in auto_topic_list if 'No Topics' not in x]
        source = [item for item, mask in zip(mydocs, select) if mask]
        file_name = './output/{}_auto_output.txt'.format(self.ric)

        _bug = [True if 'Topic: Consumer spending' in x else False for x in auto_topic_list]
        _temp = [item for item, mask in zip(mydocs, _bug) if mask]
        write_list_to_file(_temp, 'bug.txt')

        write_list_to_file(auto_topic_subset, file_name)
        contents = '\n\n'.join(auto_topic_subset)
        print('.......')
        summary = openai_completions_roles(sys_prompt={'role': 'system', 'content': topic_prompt_combine_template},
                                           human_prompt={'role':'user',
                                                         'content': human_template.replace('{text}', contents)})

        print(summary)
        summary_filename = './output/{}_auto_summary.txt'.format(self.ric)
        write_list_to_file([summary], summary_filename)
        return auto_topic_subset, source, summary



if __name__ == '__main__':
    # --------------------------------------------------------------
    # Load env
    # --------------------------------------------------------------
    load_dotenv()
    openai.api_key =os.getenv('OPENAI_API_KEY')
    # --------------------------------------------------------------
    # Call class a
    # --------------------------------------------------------------
    test = Summarizer()
    # --------------------------------------------------------------
    # Read in datafiles and return documents. (langchain class)
    # --------------------------------------------------------------

    all_docs = test.prepare_data_from_files('./data/')
    # --------------------------------------------------------------
    # Specify a company
    # --------------------------------------------------------------
    myric = 'Amazon'
    test.ric = myric
    mydocs = [x for x in all_docs if myric in x.metadata['company']]
    print(len(mydocs))

    # --------------------------------------------------------------
    # Generate topic specific summary, can change pred prompt and summary prompt.
    # --------------------------------------------------------------
    # test.topic_extract_summary(PRED_PROMPT_CONSUEMR_SPENDING, 'Consumer spending or confidence', mydocs=mydocs)
    # test.topic_extract_summary(PRED_PROMPT_LABOR_COST, 'Labor Cost')

    # --------------------------------------------------------------
    # Generate automatic full summary, can change pred prompt and summary prompt.
    # --------------------------------------------------------------
    #a,b,summary = test.doc_summary(TOPIC_PROMPT_MAP_TEMPLATE, TOPIC_PROMPT_COMBINE_TEMPLATE2, mydocs=mydocs)


