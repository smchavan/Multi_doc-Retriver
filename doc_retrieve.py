import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

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
loaders = [pdf_loader, readme_loader, txt_loader]
#lets create document 
documents = []
for loader in loaders:
    documents.extend(loader.load())

documents = loader.load()
#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))
print(texts[5])

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, 
                                embedding=embedding,
                                persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("How much money did Pando raise?")
print(len(docs))

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

print(retriever.search_type)

print(retriever.search_kwargs)

# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

# break it down
query = "What is the news about Pando?"
llm_response = qa_chain(query)
# process_llm_response(llm_response)
llm_response

query = "Who led the round in Pando?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What did databricks acquire?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is generative ai?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "Who is CMA?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

print(qa_chain.retriever.search_type , qa_chain.retriever.vectorstore)

# To cleanup, you can delete the collection
vectordb.delete_collection()
vectordb.persist()


# Link to the google colab : https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=Jl84qGQt5Wu5
