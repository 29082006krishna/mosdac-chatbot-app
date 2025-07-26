__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
import streamlit as st
import asyncio # New import for asyncio handling
import nest_asyncio # New import for asyncio handling

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit Cloud secrets.")
    st.stop()

# Apply nest_asyncio to handle potential asyncio conflicts in Streamlit's environment
nest_asyncio.apply() # <<< IMPORTANT FIX FOR EVENT LOOP ERROR

VECTOR_DB_PATH = "./mosdac_vector_db"

# --- SEO Enhancement START ---
st.set_page_config(
    page_title="MOSDAC AI Chatbot: Your Satellite Data Assistant", # Enhanced browser tab title
    page_icon="🛰️", # A relevant emoji
    layout="wide" # Optional: Makes the app wider for better viewing
)
# --- SEO Enhancement END ---


# --- SEO Enhancement START ---
st.title("🛰️ MOSDAC AI Chatbot: Your Satellite Data Assistant")
st.markdown("""
**Welcome! Ask me anything about the Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC), ISRO's earth observation missions, and related data products.**

This AI-powered chatbot provides instant, accurate, and concise information based on MOSDAC's comprehensive knowledge base. You can inquire about:
* **Satellite Missions:** Details on missions like INSAT-3DR, Oceansat-2, RISAT-1, Megha-Tropiques, and more.
* **Data Products:** Information on various levels of data (Level-1, Level-2), meteorological products, oceanographic data, and derived products.
* **Data Access & Registration:** Guidance on how to register on the MOSDAC website and access satellite imagery and other datasets.
* **Applications:** Explore how MOSDAC data is used in weather forecasting, climate monitoring, oceanography, and disaster management.

Start by asking a question like:
* "What is INSAT-3DR?"
* "How can I access satellite data from MOSDAC?"
* "Tell me about ocean data products."
* "What is MOSDAC used for?"
""")
# --- SEO Enhancement END ---


# Initialize embedding model and LLM using Streamlit's cache
# This function is run only only once thanks to @st.cache_resource,
# which helps prevent asyncio-related errors.
@st.cache_resource
def get_llm_and_embeddings(api_key):
    try:
        # Explicitly set a new event loop if one is not running (fallback for nest_asyncio)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # IMPORTANT: Updated embedding model name to "models/embedding-001"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        # Keeping "gemini-pro" for LLM as it's generally correct.
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)
        return llm, embeddings
    except Exception as e:
        st.error(f"Failed to initialize LLM or Embeddings. Check your API key and internet connection: {e}")
        st.stop()

# Call the cached function to get the LLM and embeddings instances
llm, embeddings = get_llm_and_embeddings(GOOGLE_API_KEY)


# Load the persisted vector store
if os.path.exists(VECTOR_DB_PATH):
    try:
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks
    except Exception as e:
        st.error(f"Failed to load vector database. Did you run create_vector_db.py? Error: {e}")
        st.stop()
else:
    st.error(f"Vector database not found at {VECTOR_DB_PATH}. Please run `python create_vector_db.py` first.")
    st.stop()

# Define the RAG prompt template
template = """You are a helpful and knowledgeable assistant for the MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre) website.
Your goal is to provide accurate and concise information based on the provided context.
If the context does not contain the answer, politely state that you don't have enough information from the MOSDAC knowledge base to answer.
Do not make up information.
If a URL is present in the context for a specific item, always include it in your answer.

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Build the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_input := st.chat_input("Ask about MOSDAC..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_input)
            st.markdown(response)
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("Developed with LangChain, ChromaDB, and Google Gemini.")