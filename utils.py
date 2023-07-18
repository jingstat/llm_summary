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
import re
from langchain.vectorstores import Chroma

def create_sentences(segments, MIN_WORDS, MAX_WORDS):
    # Combine the non-sentences together
    sentences = []
    is_new_sentence = True
    sentence_length = 0
    sentence_num = 0
    sentence_segments = []

    for i in range(len(segments)):
        if is_new_sentence == True:
            is_new_sentence = False
        # Append the segment
        sentence_segments.append(segments[i])
        segment_words = segments[i].split(' ')
        sentence_length += len(segment_words)

        # If exceed MAX_WORDS, then stop at the end of the segment
        # Only consider it a sentence if the length is at least MIN_WORDS
        if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
            sentence = ' '.join(sentence_segments)
            sentences.append({
                'sentence_num': sentence_num,
                'text': sentence,
                'sentence_length': sentence_length
            })
            # Reset
            is_new_sentence = True
            sentence_length = 0
            sentence_segments = []
            sentence_num += 1

    return sentences

def create_chunks(sentences, CHUNK_LENGTH, STRIDE):
    sentences_df = pd.DataFrame(sentences)

    chunks = []
    for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
        chunk = sentences_df.iloc[i:i + CHUNK_LENGTH]
        chunk_text = ' '.join(chunk['text'].tolist())

        chunks.append({
            'start_sentence_num': chunk['sentence_num'].iloc[0],
            'end_sentence_num': chunk['sentence_num'].iloc[-1],
            'text': chunk_text,
            'num_words': len(chunk_text.split(' '))
        })

    chunks_df = pd.DataFrame(chunks)
    return chunks_df.to_dict('records')

def parse_title_summary_results(results):
    out = []
    for e in results:
        e = e.replace('\n', '')
        if '|' in e:
            processed = {'title': e.split('|')[0],
                         'summary': e.split('|')[1][1:]
                         }
        elif ':' in e:
            processed = {'title': e.split(':')[0],
                         'summary': e.split(':')[1][1:]
                         }
        elif '-' in e:
            processed = {'title': e.split('-')[0],
                         'summary': e.split('-')[1][1:]
                         }
        else:
            processed = {'title': '',
                         'summary': e
                         }
        out.append(processed)
    return out

def load_text(txt_path):
    with open(txt_path, 'r') as f:
        txt = f.readlines()
    segments = [x for x in txt if x != '\n']
    pattern = re.compile(r'\(\d{2}:\d{2}\)(:)?\n')    # remove (xx:xx)
    segments = [re.sub(pattern, '', s) for s in segments]
    segments = [re.split('(?<!\d)\.(?!\d)', s) for s in segments]  #Split sentences by ., but keep number like 0.5
    segments = [item for sublist in segments for item in sublist]
    segments = [x for x in segments if x not in ['\n', '']]   # flatten list
    segments = [segment + '.' for segment in segments]   # add . to the end of stences
    return segments
