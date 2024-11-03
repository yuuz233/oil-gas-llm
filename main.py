import os
import random
from typing import Dict, Tuple, Any
import json
from pathlib import Path

import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.schema import Document
import fitz
import requests  # Import requests

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict, Any
DEV_API_KEY = "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"


import plotly.graph_objects as go

def plot_predictions(predictions):
    """Generate a Plotly graph for the model predictions."""
    days = [entry['days_on_production'] for entry in predictions]
    P50_rates = [entry['prediction']['P50_rate'] for entry in predictions]

    # Create a line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=days, y=P50_rates, mode='lines+markers', name='P50 Rate', line=dict(color='orange')))

    fig.update_layout(
        title='Reservoir Production Predictions',
        xaxis_title='Days on Production',
        yaxis_title='Production Rate (STB/day)',
        legend_title='Prediction Percentiles',
        template='plotly_white'
    )

    return fig


def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
    """
    Load a PDF file and convert each page into a LangChain Document object.
    """
    documents = []
    with fitz.open(str(pdf_path)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            documents.append(Document(
                page_content=text,
                metadata={"page": page_num + 1, "source": pdf_path.name}
            ))
    return documents


def load_pdfs_from_directory(directory: Path) -> List[Document]:
    """Load all PDFs from a directory using platform-independent paths"""
    pdf_documents = []
    for pdf_path in directory.glob("*.pdf"):
        pdf_documents.extend(load_pdf_as_documents(pdf_path))
    return pdf_documents


# Get the directory containing the script
script_dir = Path(__file__).parent.resolve()
print(f"Loading PDFs from: {script_dir}")
pdf_docs = load_pdfs_from_directory(script_dir)
print(f"Loaded {len(pdf_docs)} PDF documents")


def setup_vector_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=DEV_API_KEY)
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
4. Perform vector similarity search using InterSystems IRIS for finding related data.

Decide which action to take based on the user's question.
If the question is factual, retrieve documents. If it requires computation, run the model.
For finding similar or related information, use the IRIS vector search.
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"], template=agent_prompt)


def external_model_run(question):
    # Replace this with the actual external model logic
    return "Running external model for computational query."


def vector_search_iris(query: str) -> Dict[str, Any]:
    """
    Placeholder for InterSystems IRIS vector search implementation
    """
    # This would be replaced with actual IRIS vector search logic
    return {
        "results": [
            {"content": "Sample search result 1", "similarity": 0.95},
            {"content": "Sample search result 2", "similarity": 0.85}
        ],
        "metadata": {"source": "IRIS Vector Store"}
    }


def get_default_param_ranges() -> Dict[str, Tuple[float, float]]:
    """Define default parameter ranges"""
    return {
        "days_on_production": (0, 365),
        "depth": (5000, 12000),
        "permeability": (1, 1000),
        "porosity": (0.05, 0.35),
        "initial_pressure": (2000, 6000),
        "temperature": (100, 250),
        "thickness": (20, 200),
        "initial_water_saturation": (0.1, 0.9),
        "water_cut": (0, 0.8),
        "flowing_pressure": (500, 4000),
    }


def generate_random_value(param_range: Tuple[float, float]) -> float:
    """Generate a random value within the given range"""
    min_val, max_val = param_range
    if isinstance(min_val, int) and isinstance(max_val, int):
        return random.randint(min_val, max_val)
    return random.uniform(min_val, max_val)


def execute_model(query: str) -> str:
    """
    Execute the model with interactive parameter generation
    """
    param_ranges = get_default_param_ranges()
    generated_params = {}

    # Initialize chat
    st.write("Let's generate parameters for the model.")
    st.write("I'll generate values based on physical properties, but you can adjust them if needed.")

    # Generate and potentially modify each parameter
    for param, value_range in param_ranges.items():
        # Generate initial random value
        initial_value = generate_random_value(value_range)
        generated_params[param] = initial_value

    # Display final parameters as JSON
    st.write("🎯 Final Parameters (in JSON format):")
    st.session_state.params = (generated_params)
    st.json(st.session_state.params)    

    # Attempt to send parameters to the model server
    try:
        print("Sending parameters to model server")
        response = requests.post("http://127.0.0.1:8000/predict", json=generated_params)  # Replace with your server URL
        
        if response.status_code == 200:
            predictions = response.json()
            st.session_state.predictions = predictions  # Store predictions in session state
            st.write("### Predictions")
            
            # Plot the predictions
            fig = plot_predictions(predictions)
            st.plotly_chart(fig)
        else:
            st.error("Failed to retrieve predictions.")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error during model execution: {e}")
        return "Model execution failed."

    return "Model executed successfully."

def request_parameters_from_llm(prompt: str) -> Dict[str, float]:
    """Request parameter values from the LLM based on the user's prompt."""
    # Define the prompt for the LLM
    llm_prompt = f"Given the question: '{prompt}', suggest the best parameters for the reservoir model within the following ranges:\n"
    llm_prompt += json.dumps(get_default_param_ranges())

    # Call the LLM API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {DEV_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-3.5-turbo",  # or any other model you want to use
            "messages": [{"role": "user", "content": llm_prompt}],
            "temperature": 0.5,
        }
    )

    if response.status_code == 200:
        response_data = response.json()
        # Parse the response to extract parameters
        generated_params = json.loads(response_data['choices'][0]['message']['content'])
        return generated_params
    else:
        st.error(f"LLM request failed: {response.text}")
        return {}

# Set page config
st.set_page_config(
    page_title="LangChain Agent App",
    page_icon="🤖",
    layout="wide"
)
if 'params' in st.session_state:
    st.write("### Previous Parameters")
    st.json(st.session_state.params)
if 'predictions' in st.session_state:
    st.write("### Previous Predictions")
    fig = plot_predictions(st.session_state.predictions)
    st.plotly_chart(fig)
# Initialize session state for OpenAI API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:",
                            type="password", value=DEV_API_KEY)
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
            description="Use this tool to run external models for computational tasks, or to generate ideas for simulation runs."
        )

        iris_search_tool = Tool(
            name="IRISVectorSearch",
            func=vector_search_iris,
            description="Use this tool to perform vector similarity search in the IRIS database for finding related information and patterns."
        )

        model_execution_tool = Tool(
            name="ExecuteModel",
            func=execute_model,
            description="Use this tool to execute the model with interactive parameter generation. "
                        "It will help generate and customize model parameters through conversation."
        )

        # Initialize the agent
        agent = initialize_agent(
            llm=llm,
            tools=[retrieval_tool, model_execution_tool, iris_search_tool],
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
