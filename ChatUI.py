import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import os
import sys
import openai
from PyPDF2 import PdfReader
from cprint import cprint
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.vectorstores import Chroma
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
import DLAIUtils
from langchain.retrievers.multi_query import MultiQueryRetriever


import pandas as pd

LOCAL_VAR = DLAIUtils.Utils()
OPENAI_API_KEY = LOCAL_VAR.get_openai_api_key()
PINECONE_API_KEY = LOCAL_VAR.get_pinecone_api_key()
INDEX_NAME = DLAIUtils.INDEX_NAME
ENVIRONMENT = DLAIUtils.ENVIRONMENT



st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:", layout="wide")
st.header("Chat with multiple PDFs :books:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello!! I am SalesOps Bot. How can I help you ?")
    ]

with st.sidebar:
    st.header("Documents")
    df = pd.DataFrame({"Doc Name":("FY24_Sales_Compensation_FAQ.pdf", "RRT FAQs 12112023.pdf")})
    st.data_editor(df,hide_index=True)


# location of the pdf file/files.
loader = DirectoryLoader('./documents/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len, )
texts = text_splitter.split_documents(documents)
print(f"# of Document chunks: {len(texts)}")


embeddings = OpenAIEmbeddings()
#vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
#vectordb.persist()
#vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
#retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={"k": 4})


pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
#to load document chunks to a new index first time use below
#vectordb = PineconeStore.from_documents(documents=texts, embedding=embeddings, index_name=index_name)
## to retrieve using existing index use below
vectordb = PineconeVectorStore(pinecone_api_key=os.environ['PINECONE_API_KEY'], index_name=INDEX_NAME,embedding=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})



template = """You are friendly chatbot helping the operations, support and sales agents teams and your name is SalesOps Bot. 
Use the following pieces of context to provide a detailed answer to the question.If you don't know the answer, just say that you don't 
know it, do not try to make up an answer.

<context>
{context}
</context>

Question: {question}
Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough(), }
    | prompt
    | model
    | StrOutputParser()
)

def get_response(user_question):
    ai_response = chain.invoke(user_question)
    return ai_response

#user input
user_question = st.chat_input("Type your messages here...")

if user_question is not None and user_question != "":
    response = get_response(user_question)
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=response))
    # print('-' * 100)
    # print(f"{retriever.get_relevant_documents(user_question)[0]}")
    # print('-' * 100)
    with open("RecordContext.txt", mode="a") as data:
        docs = retriever.get_relevant_documents(user_question)
        for i in docs:
            data.write(f"\nQuestion is : {user_question}\nContext retrieved : {i.page_content}\n\n")
            data.write('--'*50)

# with st.sidebar:
#     st.write(st.session_state.chat_history)

#conversation
for messages in st.session_state.chat_history:
    if isinstance(messages,AIMessage):
        with st.chat_message("AI"):
            st.write(messages.content)
    elif isinstance(messages,HumanMessage):
        with st.chat_message("Human"):
            st.write(messages.content)












