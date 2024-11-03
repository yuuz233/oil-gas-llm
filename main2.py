import os

import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.schema import Document
import fitz
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
DEV_API_KEY = "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"


def load_pdf_as_documents(pdf_path):
    """
    Load a PDF file and convert each page into a LangChain Document object.
    """
    documents = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            documents.append(Document(page_content=text,
                             metadata={"page": page_num + 1}))
    return documents


def load_pdfs_from_directory(directory):
    pdf_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            pdf_documents.extend(load_pdf_as_documents(pdf_path))
    return pdf_documents


pdf_directory = "documents"
print("Loading pdf")
pdf_docs = load_pdfs_from_directory(pdf_directory)
print("pdf loaded")


def setup_vector_store(docs):
    embeddings = OpenAIEmbeddings(
        openai_api_key="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")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store.as_retriever()


retriever = setup_vector_store(pdf_docs)

agent_prompt = """
You are an assistant that can either:
1. Retrieve documents to answer factual questions.
2. Run a surrogate model for tasks requiring computation.
3. Run a simulation model for tasks requiring computation.

Decide which action to take based on the user's question.
If the question is factual, retrieve documents. If it requires computation, run the model.
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"], template=agent_prompt)


def external_model_run(question):
    # Replace this with the actual external model logic
    return "Running external model for computational query."


# Set page config
st.set_page_config(
    page_title="LangChain Agent App",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for OpenAI API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state.openai_api_key = api_key

# Main content
st.title("🤖 LangChain Agent Interface")

if False:
    # if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    try:
        # Initialize LangChain components
        llm = OpenAI(temperature=0.7, openai_api_key=DEV_API_KEY)

        # Define the RetrievalQA chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever)
        # tools = load_tools(["wikipedia", "llm-math"], llm=llm)
        memory = ConversationBufferMemory(memory_key="chat_history")

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

        # Initialize the agent
        agent = initialize_agent(
            llm=llm,
            tools=[retrieval_tool, external_model_tool],
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = agent.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning(
            "If you're seeing an API error, please check if your OpenAI API key is valid.")

# Add helpful information
st.sidebar.markdown("""
### About
This app uses LangChain's agent framework to:
- Process natural language queries
- Access tools like Wikipedia and math calculations
- Maintain conversation memory
""")
