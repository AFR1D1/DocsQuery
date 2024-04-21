import os
import shutil
from google.colab import files
import sys
import subprocess
import tempfile
import time



# Dynamically importing libraries not directly related to Whoosh (handling these as you did originally)
library_names = ['whoosh','langchain', 'langchain-openai', 'PyPDF2', 'python-docx', 'openai', 'tiktoken', 'python-pptx', 'textwrap']
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])

import docx
import pptx
from PyPDF2 import PdfReader 
import textwrap
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from whoosh import qparser
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from getpass import getpass

if "OPENAI_API_KEY" in os.environ:
    print("Token already set.")
else:
    token = getpass("Enter your OpenAI token: ")
    os.environ["OPENAI_API_KEY"] = str(token)

# Whoosh does not need a specific embeddings object
chain = load_qa_chain(OpenAI(), chain_type="stuff")



def create_index(index_dir):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    schema = Schema(content=TEXT(stored=True))
    return create_in(index_dir, schema)

def add_documents_to_index(indexer, texts):
    writer = indexer.writer()
    for text in texts:
        writer.add_document(content=text)
    writer.commit()

def search_index(query, indexer):
    with indexer.searcher() as searcher:
        query_parser = QueryParser("content", indexer.schema)
        query_obj = query_parser.parse(query)
        results = searcher.search(query_obj)
        return [hit['content'] for hit in results]

def extract_and_index_texts(root_files, index_dir):
    raw_text = ''
    for root_file in root_files:
        _, ext = os.path.splitext(root_file)
        if ext == '.pdf':
            with open(root_file, 'rb') as f:
                reader = PdfReader(f)
                raw_text += ' '.join([page.extract_text() for page in reader.pages])
        elif ext == '.docx':
            doc = docx.Document(root_file)
            raw_text += ' '.join([p.text for p in doc.paragraphs])
        elif ext == '.pptx':
            ppt = pptx.Presentation(root_file)
            raw_text += ' '.join([shape.text for slide in ppt.slides for shape in slide.shapes if hasattr(shape, 'text')])
    
    texts = textwrap.wrap(raw_text, 1000)
    indexer = create_index(index_dir)
    add_documents_to_index(indexer, texts)
    return indexer

def run_conversation(folder_path):
    index_dir = os.path.join(folder_path, 'indexdir')
    root_files = upload_file(folder_path)
    indexer = extract_and_index_texts(root_files, index_dir)

    count = 0
    while True:
        print("Question ", count + 1)
        query = input(" Ask questions or type stop:\n ")
        
        if query.lower() == "stop":
            print("Thanks.")
            break
        elif query == "":
            print("Input is empty!")
            continue
        else:
            docs = search_index(query, indexer)
            response = chain.run(input_documents=docs, question=query)
            wrapped_text = textwrap.wrap(response, width=100)
            print("Answer:")
            for line in wrapped_text:
                print(line)
            count += 1

def upload_file(folder_path):
    uploaded = files.upload()
    root_file = []

    for filename, data in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(data)
        shutil.copy(filename, folder_path + "/")
        root_file.append(folder_path + "/" + filename)
        os.remove(filename)

    return root_file

