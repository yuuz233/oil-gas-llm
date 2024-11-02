import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory

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

if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    try:
        # Initialize LangChain components
        llm = OpenAI(
            temperature=0, openai_api_key=st.session_state.openai_api_key)
        tools = load_tools(["wikipedia", "llm-math"], llm=llm)
        memory = ConversationBufferMemory(memory_key="chat_history")

        # Initialize the agent
        agent = initialize_agent(
            tools,
            llm,
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
