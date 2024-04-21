import sys
import subprocess
from google.colab import files
import tempfile
import shutil
import os
import time



library_names = ['whoosh','langchain', 'langchain-openai', 'faiss-cpu', 'PyPDF2','python-docx', 'openai', 'tiktoken', 'python-pptx', 'textwrap', ]

# Dynamically importing libraries
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from PyPDF2 import PdfReader 
import textwrap
import docx
import pptx
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from getpass import getpass




#token adding
if "OPENAI_API_KEY" in os.environ:
    print("Token already set.")
else:
    token = getpass("Enter your OpenAI token: ")
    os.environ["OPENAI_API_KEY"] = str(token)





# Downloading embeddings from OpenAI
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def create_search_index(texts, index_dir):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    schema = Schema(content=TEXT(stored=True), path=ID(stored=True, unique=True))
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for idx, text in enumerate(texts):
        writer.add_document(content=text, path=str(idx))
    writer.commit()
    return ix

def extract_texts(root_files, index_dir):
    raw_text = ''
    for root_file in root_files:
        _, ext = os.path.splitext(root_file)
        if ext == '.pdf':
            with open(root_file, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    raw_text += page.extract_text() or ''
        elif ext == '.docx':
            doc = docx.Document(root_file)
            for paragraph in doc.paragraphs:
                raw_text += paragraph.text + '\n'
        elif ext == '.pptx':
            ppt = pptx.Presentation(root_file)
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, 'text'):
                        raw_text += shape.text + '\n'

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return create_search_index(texts, index_dir)


def upload_file(folder_path):
    uploaded = files.upload()
    root_files = []
    for filename, data in uploaded.items():
        local_path = os.path.join(tempfile.gettempdir(), filename)
        with open(local_path, 'wb') as f:
            f.write(data)
        shutil.move(local_path, folder_path)
        root_files.append(os.path.join(folder_path, filename))
    return root_files


def run_conversation(folder_path):
    root_files = upload_file(folder_path)
    index_dir = os.path.join(folder_path, "indexdir")
    docsearch = extract_texts(root_files, index_dir)
    count = 0
    while True:
        query = input(f"Question {count + 1} (type 'stop' to exit): ")
        if query.lower() == "stop":
            print("Exiting. Thank you for using DocsQuery.")
            break
        elif not query.strip():
            print("Input is empty, please ask a question.")
            continue
        response = run_query(query, docsearch)
        wrapped_text = textwrap.wrap(response, width=100)
        print("\nAnswer:")
        for line in wrapped_text:
            print(line)
        count += 1
