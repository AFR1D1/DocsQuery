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
import openai

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
    # Creating a set of indices for tokens that should be excluded (URLs in this case)
    excluded_tokens = {start for match_id, start, end in matches}
    keywords = [
        token.lemma_ for token in doc 
        if token.pos_ in {'NOUN', 'PROPN', 'VERB'} 
        and not token.is_stop 
        and token.i not in excluded_tokens  # Exclude tokens that are part of a URL
    ]
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

    # Return both the FAISS index and the texts
    return docsearch, texts


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


def find_similar_questions(query, questions, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [questions[index] for index in top_results.indices]

from openai import OpenAI

# Instantiate the client with your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_questions(client, keywords, max_questions=5):
    """
    Generates a list of questions for each keyword using the OpenAI language model.
    Parameters:
    client: An OpenAI client instance.
    keywords: A set of extracted keywords.
    max_questions: Maximum number of questions to generate per keyword.
    Returns:
    A list of questions.
    """
    questions = []
    for keyword in keywords:
        # Use the chat completion method of the client for question generation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Generate {max_questions} questions about the keyword '{keyword}':" }
            ],
        )
        # Extract generated messages and append to questions list
        for message in response['choices'][0]['message']:
            if message['role'] == 'assistant':  # We want to capture only the assistant's messages
                questions.append(message['content'])
    return questions

def load_all_questions(all_texts, model):
    """
    Generates questions based on the content of uploaded files dynamically using a language model.
    Parameters:
    all_texts: A list of strings containing the text segments of the uploaded files.
    model: An instance of the OpenAI API model for generating questions.
    Returns:
    A list of generated questions based on the keywords extracted from the text.
    """
    unique_keywords = set()
    for text in all_texts:
        keywords = extract_keywords(text)
        unique_keywords.update(keywords)
    
    # Generate questions dynamically from keywords using a language model
    all_questions = generate_questions(client, unique_keywords)
    return all_questions


#################



from openai import OpenAI

def run_conversation(folder_path):
    # Instantiate the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    root_files = upload_file(folder_path)
    docsearch, all_texts = extract_texts(root_files)  # Receive texts as well
    all_questions = load_all_questions(all_texts, client)  # Pass the client instance here
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
            keywords = extract_keywords(query)
            
            # Run the query against the documents
            response = run_query(query, docsearch)
            
            # Print the response
            wrapped_text = textwrap.wrap(response, width=100)
            print("Answer:")
            for line in wrapped_text:
                print(line)
            
            # Generate and print related questions suggestions
            suggestions = find_similar_questions(query, all_questions)
            print("Related questions:")
            for question in suggestions:
                print(question)
            
            count += 1


