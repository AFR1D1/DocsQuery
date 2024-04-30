import sys
import subprocess
from google.colab import files
import tempfile
import shutil
import os
import time



library_names = ['spacy','pytesseract', 'sentence-transformers', 'langchain', 'langchain-openai', 'faiss-cpu', 'PyPDF2','python-docx', 'openai', 'tiktoken', 'python-pptx', 'textwrap', ]

# Dynamically importing libraries
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])

from pytesseract import image_to_string
from PIL import Image
from PyPDF2 import PdfReader 
import textwrap
import docx
import pptx
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from getpass import getpass
import io

# Import spaCy after installation
import spacy
from spacy.matcher import Matcher

# Downloading the English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("en_core_web_sm not found. Downloading en_core_web_sm...")
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)

# Add match pattern for URLs
pattern = [{"LIKE_URL": True}]
matcher.add("URL_PATTERN", [pattern])

def extract_keywords(text):
    doc = nlp(text)
    matches = matcher(doc)
    excluded_tokens = {start for match_id, start, end in matches}
    keywords = [token.lemma_ for token in doc if token.pos_ in {'NOUN', 'PROPN', 'VERB'} and not token.is_stop and token.i not in excluded_tokens]
    return keywords




#token adding
if "OPENAI_API_KEY" in os.environ:
    print("Token already set.")
else:
    token = getpass("Enter your OpenAI token: ")
    os.environ["OPENAI_API_KEY"] = str(token)





# Downloading embeddings from OpenAI
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def generate_questions(keywords):
    questions = []
    for keyword in set(keywords):
        questions.extend([
            f"What is {keyword}?",
            f"How does {keyword} work?",
            f"What are the applications of {keyword}?",
            f"Explain the concept of {keyword}",
            f"Advantages and disadvantages of {keyword}?"
        ])
    return questions

def extract_texts(root_files):
    raw_text = ''
    for root_file in root_files:
        _, ext = os.path.splitext(root_file)
        if ext == '.pdf':
            with open(root_file, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + '\n'
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
    return raw_text

def run_query(query, docsearch):
    """
    Executes a search query on a PDF file utilizing the docsearch and chain libraries.
    Parameters:
    query: A string that specifies the query to be executed.
    file: A PDFReader object that holds the PDF file to be queried.
    Returns:
    A string that includes the results from applying the chain library to the documents retrieved by the docsearch similarity search.
    """
    
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)


def upload_file(folder_path):
    """
    Uploads a file from the local file system and stores it in a specified directory.
    Parameters:
    folder_path: A string that indicates the directory where the file should be saved.
    Returns:
    A string that denotes the path of the file that has been uploaded.
    """
    
    uploaded = files.upload()
    root_file = []

    for filename, data in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(data)
        shutil.copy(filename, folder_path + "/")
        root_file.append(folder_path + "/" + filename)
        os.remove(filename)


    return root_file
################
import spacy
import torch
from sentence_transformers import SentenceTransformer, util



nlp = spacy.load("en_core_web_sm")  # Load a language model
model = SentenceTransformer('all-MiniLM-L6-v2')






##############


def run_conversation(folder_path):
    root_files = upload_file(folder_path)
    all_texts = extract_texts(root_files)
    keywords = extract_keywords(all_texts)
    all_questions = generate_questions(keywords)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts([all_texts], embeddings)

    count = 0
    while True:
        print(f"Question {count + 1}")
        query = input("Ask questions or type stop:\n")
        if query.lower() == "stop":
            break
        elif query == "":
            print("Input is empty!")
            continue

        response = run_query(query, docsearch)
        print("Answer:", textwrap.fill(response, width=100))
        suggestions = find_similar_questions(query, all_questions)
        print("Related questions:")
        for question in suggestions:
            print(question)
        count += 1

def run_query(query, docsearch):
    docs = docsearch.similarity_search(query)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

def find_similar_questions(query, questions, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [questions[index] for index in top_results.indices]

if __name__ == "__main__":
    folder_path = tempfile.mkdtemp()
    run_conversation(folder_path)
