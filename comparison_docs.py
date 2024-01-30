import streamlit as st
import os
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
import openai
os.environ["OPENAI_API_KEY"] = ""
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-03-15-preview'
os.environ['OPENAI_API_BASE'] = "https://summarization"

#API settings for embedding
openai.api_type = "azure"
openai.api_base = "https://summarization"
openai.api_version = '2023-03-15-'
openai.api_key = ""
    
    
class DocumentInput(BaseModel):
    question: str = Field()

# Create a temporary directory in the script's folder
script_dir = Path(__file__).resolve().parent
temp_dir = os.path.join(script_dir, "tempDir")


def main():
    st.title("PDF Document Comparison")

    # Create a form to upload PDF files and enter a question
    st.write("Upload the first PDF file:")
    pdf1 = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf1")

    st.write("Upload the second PDF file:")
    pdf2 = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf2")

    question = st.text_input("Enter your question")
    submit_button = st.button("Compare PDFs")

    if submit_button:
        if pdf1 and pdf2:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            else:
                # Clear the previous contents of the "tempDir" folder
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting file: {e}")

            # Save the PDF files to the "tempDir" directory
            pdf1_path = os.path.join(temp_dir, pdf1.name)
            with open(pdf1_path, 'wb') as f:
                f.write(pdf1.getbuffer())

            pdf2_path = os.path.join(temp_dir, pdf2.name)
            with open(pdf2_path, 'wb') as f:
                f.write(pdf2.getbuffer())



            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",engine="gpt-35-turbo")

            tools = []
            files = [

                {
                    "name": pdf1.name,
                    "path": pdf1_path,
                },

                {
                    "name": pdf2.name,
                    "path": pdf2_path,
                },
            ]

            for file in files:
                loader = PyPDFLoader(file["path"])
                pages = loader.load_and_split()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(pages)
                embeddings = OpenAIEmbeddings()
                retriever = FAISS.from_documents(docs, embeddings).as_retriever()

                # Wrap retrievers in a Tool
                tools.append(
                    Tool(
                        args_schema=DocumentInput,
                        name=file["name"],
                        description=f"useful when you want to answer questions about {file['name']}",
                        func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                    )
                )
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                verbose=True,
            )

            st.write(agent({"input": question}))
            # Now you have both PDFs saved in the "tempDir" folder
            # You can perform your PDF comparison here


if __name__ == "__main__":
    main()