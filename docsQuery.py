import sys
import subprocess
from google.colab import files
import tempfile
import shutil
import os
import time



library_names = ['langchain', 'langchain-openai', 'faiss-cpu', 'PyPDF2','python-docx', 'openai', 'tiktoken', 'python-pptx', 'textwrap', ]

# Dynamically importing libraries
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])


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
            with open(root_file, 'rb') as f:
                reader = PdfReader(f)
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    raw_text += page.extract_text()
        elif ext == '.docx':
            doc = docx.Document(root_file)
            for paragraph in doc.paragraphs:
                raw_text += paragraph.text
        elif ext == '.pptx':
            ppt = pptx.Presentation(root_file)
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, 'text'):
                        raw_text += shape.text

    # retreival we don't hit the token size limits. 
    text_splitter = CharacterTextSplitter(        
                                            separator = "\n",
                                            chunk_size = 1000,
                                            chunk_overlap  = 200,
                                            length_function = len,
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


def run_conversation(folder_path):
    """
    Starts a dialogue with the user by continuously requesting input queries and processing them against a PDF file.
    Parameters:
    folder_path: A string that specifies the location of the folder containing the PDF file.
    Returns:
    Conducts a conversation based on the PDF file.
    """
    root_files = upload_file(folder_path)
    # location of the pdf


    docsearch = extract_texts(root_files)

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
            wrapped_text = textwrap.wrap(run_query(query, docsearch), width=100)
            print("Answer:")
            for line in wrapped_text:
                print(line)
            count += 1
