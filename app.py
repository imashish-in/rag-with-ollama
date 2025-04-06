import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from vector_db_manager import MODEL_NAME
import time
from functools import lru_cache

# Set page config
st.set_page_config(
    page_title="Document RAG with Llama 3.2",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f0f0;
    }
    .chat-message.assistant {
        background-color: #e6e6e6;
    }
    .chat-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .user-label {
        color: #1f77b4;
    }
    .assistant-label {
        color: #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'initialization_complete' not in st.session_state:
    st.session_state.initialization_complete = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model=MODEL_NAME)

@lru_cache(maxsize=1)
def create_qa_chain(vectorstore):
    # Create QA chain
    llm = Ollama(model=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 15  # Increased from 10 to 15 for better retrieval
            }
        ),
        return_source_documents=True
    )
    
    return qa_chain

def initialize_vectorstore():
    """Initialize the vectorstore"""
    try:
        # Load the existing vector database
        client = chromadb.PersistentClient(path="./chroma_db")
        vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=OllamaEmbeddings(model=MODEL_NAME)
        )
        
        # Verify the vector store
        try:
            # Try a simple similarity search to verify the vector store
            results = vectorstore.similarity_search("test", k=1)
            print(f"Vector store verification successful. Found {len(results)} results.")
        except Exception as e:
            print(f"Warning: Vector store verification failed: {str(e)}")
        
        return vectorstore, None
    except Exception as e:
        return None, str(e)

def process_question(question):
    """Process a question"""
    try:
        # Enhance the query to improve retrieval
        enhanced_query = f"Find information about: {question}"
        
        # Get response
        result = st.session_state.qa_chain({"query": enhanced_query})
        answer = result["result"]
        
        # If the answer indicates no information was found, provide a more helpful response
        if "don't know" in answer.lower() or "no information" in answer.lower() or "not mentioned" in answer.lower():
            return f"I couldn't find specific information about '{question}' in the available documents. The vector database might not contain information about this topic. You might want to add more relevant URLs to the urls.txt file and run the vector_db_manager.py script again.", None
        
        # Return just the answer without source information
        return answer, None
    except Exception as e:
        return None, str(e)

def initialize_rag_system():
    """Initialize the RAG system by loading the existing vector database and creating the QA chain"""
    st.session_state.is_processing = True
    
    # Create a progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update status
    status_text.text("Loading vector database...")
    
    # Initialize vectorstore
    vectorstore, error = initialize_vectorstore()
    
    if error:
        st.error(f"Failed to load the vector database: {error}")
        st.error("Please make sure you have run vector_db_manager.py first to create the vector database.")
        st.session_state.is_processing = False
        return
    
    # Store the vectorstore in session state
    st.session_state.vectorstore = vectorstore
    
    # Create QA chain
    status_text.text("Creating QA chain...")
    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
    
    status_text.text("Done! You can now ask questions about the content.")
    st.success("Vector database loaded successfully!")
    
    st.session_state.is_processing = False
    st.session_state.initialization_complete = True

# Main UI
st.title("🤖 Document RAG with Llama 3.2")
st.markdown("""
    This application allows you to:
    1. Ask questions about the content from the processed URLs
    2. Get AI-generated responses using Llama 3.2
    
    Note: Make sure you have run vector_db_manager.py first to create the vector database.
""")

# Initialize the RAG system if not already done
if not st.session_state.initialization_complete and not st.session_state.is_processing:
    with st.spinner("Initializing RAG system..."):
        initialize_rag_system()

# Chat interface
if st.session_state.qa_chain:
    # Create a container for the chat interface at the bottom
    chat_container = st.container()
    
    # Add a separator before the chat interface
    st.markdown("---")
    
    # Create a fixed-height container for the chat history
    with chat_container:
        st.markdown("### 💬 Chat with the documents")
        
        # Create a container with fixed height for chat history
        chat_history_container = st.container()
        with chat_history_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.container():
                    label_class = "user-label" if message['role'] == "user" else "assistant-label"
                    label_text = "You:" if message['role'] == "user" else "Assistant:"
                    st.markdown(f"""
                        <div class="chat-message {message['role']}">
                            <div class="chat-label {label_class}">{label_text}</div>
                            <div>{message['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Create a form for the input to ensure it's cleared after submission
        with st.form(key="chat_form", clear_on_submit=True):
            question = st.text_input("Ask a question about the documents:", key="question")
            submit_button = st.form_submit_button("Send")
        
        # Process the question if the form was submitted
        if submit_button and question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Set processing flag
            st.session_state.is_processing = True
            
            # Process question
            with st.spinner("Thinking..."):
                answer, error = process_question(question)
                
                if error:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {error}"})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Update last question
                st.session_state.last_question = question
                
                # Set processing flag to false
                st.session_state.is_processing = False
                
                # Rerun once to update the UI
                st.experimental_rerun()

else:
    if st.session_state.is_processing:
        st.info("Loading the vector database. Please wait...")
    else:
        st.error("Failed to initialize the RAG system. Please check the logs for errors.") 