import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Function to load variables from .env file
def load_env(file_path=".env"):
    with open(file_path) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Load environment variables from .env
load_env()
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Check if the API key is set
if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please check your .env file.")
if not pinecone_api_key:
    raise ValueError("Pinecone API key is not set. Please check your .env file.")

# Now you can use the retrieved API key in your code
OpenAI.api_key = openai_api_key
OpenAIEmbeddings.api_key = openai_api_key
#pc = Pinecone(api_key=pinecone_api_key)

pdf_loader = DirectoryLoader('/Users/supriya/AI/Multi_Doc Retriever/Data', glob="**/*.pdf")
readme_loader = DirectoryLoader('/Users/supriya/AI/Multi_Doc Retriever/Data', glob="**/*.md")
txt_loader = DirectoryLoader('/Users/supriya/AI/Multi_Doc Retriever/Data/new_articles', glob="**/*.txt",loader_cls=TextLoader)

#take all the loader
loader = pdf_loader
#lets create document 
documents = loader.load()
print(len(documents))

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(len(texts))
print(texts[4])






#https://colab.research.google.com/drive/17eByD88swEphf-1fvNOjf_C79k0h2DgF?usp=sharing#scrollTo=3__nT0D4Fkmg

# Load the Word documents
# doc1_path = "/Users/supriya/AI/test-langchain/Data/Test-Data1.docx"
# doc2_path = "/Users/supriya/AI/test-langchain/Data/Test-Data2.docx"

# loader = Docx2txtLoader(doc1_path)
# doc1 = loader.load()

# loader = Docx2txtLoader(doc2_path)
# doc2 = loader.load()


# # Extract the text from the documents
# doc1_text = doc1.page_content
# doc2_text = doc2.page_content
# # Split the documents into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# doc1_chunks = text_splitter.split_text(doc1_text)
# doc2_chunks = text_splitter.split_text(doc2_text)

# # Create embeddings for the document chunks
# embeddings = OpenAIEmbeddings()
# doc1_embeddings = [embeddings.embed(chunk) for chunk in doc1_chunks]
# doc2_embeddings = [embeddings.embed(chunk) for chunk in doc2_chunks]

# # Compare the embeddings
# retriever = FAISS.from_embeddings(doc1_embeddings)
# qa_chain = RetrievalQA.from_retriever(retriever)
# differences = []
# for i, emb in enumerate(doc2_embeddings):
#     result = qa_chain({"question": emb})
#     if result["result"] != "Match":
#         differences.append((i, result["result"]))

# # Display the differences
# for diff in differences:
#     chunk_index, difference = diff
#     print(f"Difference found in chunk {chunk_index}: {difference}")
