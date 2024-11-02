import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.tools import Tool
import os

# Initialize LangChain LLM
llm = OpenAI(temperature=0.7, openai_api_key="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")

# Set up the document retriever (using FAISS in this example)


def setup_vector_store(docs):
    embeddings = OpenAIEmbeddings(
        openai_api_key="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")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store.as_retriever()


# Example documents for retrieval
'''
docs = [
    {"text": "LangChain enables you to build applications with LLMs."},
    {"text": "Retrieval-Augmented Generation combines retrieval with generation for accurate answers."},
    # Add more documents as needed
]
'''
docs = [
    {"text": "abc"},
]
documents = [Document(page_content=doc["text"], metadata={}) for doc in docs]

# Initialize the vector store retriever
retriever = setup_vector_store(documents)

# Define the agent with a decision-making prompt
agent_prompt = """
You are an assistant that can either:
1. Retrieve documents to answer factual questions.
2. Run a custom model for tasks requiring computation.

Decide which action to take based on the user's question.
If the question is factual, retrieve documents. If it requires computation, run the model.
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"], template=agent_prompt)

# Define the RetrievalQA chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever)

# Custom model function (for demonstration)


def external_model_run(question):
    # Replace this with the actual external model logic
    return "Running external model for computational query."


# Streamlit UI
st.title("Question Answering with RAG and LangChain Agent")

user_question = st.text_input("Ask a question:")

retrieval_tool = Tool(
    name="RetrieveDocuments",
    func=retrieval_chain,
    description="Use this tool for answering factual questions based on document retrieval."
)

external_model_tool = Tool(
    name="RunExternalModel",
    func=external_model_run,
    description="Use this tool to run external models for computational tasks."
)

if user_question:
    # Initialize the agent
    agent = initialize_agent(
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        tools=[
            retrieval_tool,
            external_model_tool,
        ],
        verbose=True
    )

    # Decide whether to retrieve documents or run the model
    response = agent({"input": user_question})

    # Display the result
    st.write("Answer:", response['output'])
