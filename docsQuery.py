import sys
import subprocess
from google.colab import files
import tempfile
import shutil
import os
import time



library_names = ['pytesseract', 'sentence-transformers', 'langchain', 'langchain-openai', 'faiss-cpu', 'PyPDF2','python-docx', 'openai', 'tiktoken', 'python-pptx', 'textwrap', ]

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



#token adding
if "OPENAI_API_KEY" in os.environ:
    print("Token already set.")
else:
    token = getpass("Enter your OpenAI token: ")
    os.environ["OPENAI_API_KEY"] = str(token)





# Downloading embeddings from OpenAI
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def extract_texts(root_files):
    """
    Text extractes from file and puts it in a list.
    Supported file formats include: .pdf, .docx, .pptx
    If multiple files are uploaded, contents will be merged together
    Parameters:
    - root_files: A list containing the paths of the files to be processed.
    Returns:
    - A FAISS index object that includes the embeddings of the extracted text segments.
    """
    raw_text = ''

    for root_file in root_files:
        _, ext = os.path.splitext(root_file)
        if ext == '.pdf':
            # First try to extract text normally
            with open(root_file, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + '\n'
                    else:
                        # If normal text extraction doesn't work, use OCR
                        images = convert_from_path(root_file)
                        for image in images:
                            raw_text += image_to_string(image) + '\n'
        elif ext == '.docx':
            doc = docx.Document(root_file)
            for paragraph in doc.paragraphs:
                raw_text += paragraph.text + '\n'
            for rel in doc.part.rels.values():
                if 'image' in rel.reltype:
                    image_stream = io.BytesIO(rel.target_part.blob)
                    image = Image.open(image_stream)
                    raw_text += image_to_string(image) + '\n'
        elif ext == '.pptx':
            ppt = pptx.Presentation(root_file)
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, 'text'):
                        raw_text += shape.text + '\n'
                    elif shape.shape_type == 13: # ShapeType 13 corresponds to a picture
                        image_stream = io.BytesIO(shape.image.blob)
                        image = Image.open(image_stream)
                        raw_text += image_to_string(image) + '\n'

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


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

def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in {'NOUN', 'PROPN', 'VERB'} and not token.is_stop]

keywords = extract_keywords(user_query)

##############


def find_similar_questions(query, questions, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [questions[index] for index in top_results.indices]

def load_all_questions():
    # This should load or define all potential questions. Placeholder for actual implementation.
    return ["What is the best strategy?", "How to implement an algorithm?", "Examples of data structures?", "Define machine learning models", "Latest trends in technology?"]

#################

def run_conversation(folder_path):
    root_files = upload_file(folder_path)
    docsearch = extract_texts(root_files)
    all_questions = load_all_questions()
    count = 0
    while True:
        print(f"Question {count + 1}")
        query = input("Ask questions or type stop:\n")
        if query.lower() == "stop":
            print("Thanks.")
            break
        elif query == "":
            print("Input is empty!")
            continue
        else:
            # Extract keywords from the user's query
            keywords = extract_keywords(user_query)
            
            # Run the query against the documents
            response = run_query(user_query, docsearch)
            
            # Print the response
            wrapped_text = textwrap.wrap(response, width=100)
            print("Answer:")
            for line in wrapped_text:
                print(line)
            
            # Generate and print related questions suggestions
            suggestions = find_similar_questions(user_query, all_questions)
            print("Related questions:")
            for question in suggestions:
                print(question)
            
            count += 1
