import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import chromadb
import os
import shutil
from urllib.parse import urljoin, urlparse
import time

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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'urls_loaded' not in st.session_state:
    st.session_state.urls_loaded = False
if 'initialization_complete' not in st.session_state:
    st.session_state.initialization_complete = False

def clear_vector_db():
    """Clear the vector database directory"""
    db_path = "./chroma_db"
    
    # First, try to close any open ChromaDB connections
    try:
        # Force garbage collection to close connections
        import gc
        gc.collect()
    except Exception as e:
        st.warning(f"Warning during garbage collection: {str(e)}")
    
    # Add a small delay to ensure files are properly closed
    time.sleep(1)
    
    # Now try to remove the directory
    if os.path.exists(db_path):
        try:
            # Use a more direct approach to remove the directory
            import subprocess
            
            # On macOS/Linux, use rm -rf
            if os.name != 'nt':  # Not Windows
                subprocess.run(['rm', '-rf', db_path], check=True)
            else:
                # On Windows, use rmdir /s /q
                subprocess.run(['rmdir', '/s', '/q', db_path], check=True)
                
            st.info("Vector database cleared successfully using system command.")
        except Exception as e:
            st.warning(f"Could not remove directory using system command: {str(e)}")
            
            # Fallback to Python's os.remove for individual files
            try:
                for root, dirs, files in os.walk(db_path, topdown=False):
                    for name in files:
                        try:
                            file_path = os.path.join(root, name)
                            os.remove(file_path)
                        except Exception as e:
                            st.warning(f"Could not remove file {file_path}: {str(e)}")
                    
                    for name in dirs:
                        try:
                            dir_path = os.path.join(root, name)
                            os.rmdir(dir_path)
                        except Exception as e:
                            st.warning(f"Could not remove directory {dir_path}: {str(e)}")
                
                # Finally try to remove the main directory
                os.rmdir(db_path)
                st.info("Vector database cleared successfully using Python's os.remove.")
            except Exception as e:
                st.warning(f"Could not completely remove directory {db_path}: {str(e)}")
    
    # Create a fresh directory
    os.makedirs(db_path, exist_ok=True)

def get_internal_links(url, base_url):
    """Extract internal links from a webpage"""
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Skipping URL {url} - Status code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        internal_links = set()
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            
            # Only include links from the same domain
            if urlparse(absolute_url).netloc == base_domain:
                internal_links.add(absolute_url)
        
        return list(internal_links)
    except Exception as e:
        st.warning(f"Error extracting internal links from {url}: {str(e)}")
        return []

def fetch_url_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Skipping URL {url} - Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

def process_text(text, url):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    # Add URL metadata to each chunk
    chunks_with_metadata = [f"Source: {url}\n\n{chunk}" for chunk in chunks]
    
    return chunks_with_metadata

def create_vector_store(chunks_list):
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    # Flatten the list of chunks
    all_chunks = [chunk for chunks in chunks_list for chunk in chunks]
    
    # Ensure the database directory exists
    db_path = "./chroma_db"
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB with persistent client
    client = chromadb.PersistentClient(path=db_path)
    
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        client=client,
        collection_name="document_collection"
    )
    
    return vectorstore

def create_qa_chain(vectorstore):
    # Create QA chain
    llm = Ollama(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def load_urls_from_file():
    try:
        with open("urls.txt", "r") as file:
            urls = [line.strip() for line in file.readlines() if line.strip()]
        return urls
    except Exception as e:
        st.error(f"Error reading URLs file: {str(e)}")
        return []

def initialize_rag_system():
    """Initialize the RAG system by processing all URLs and creating the QA chain"""
    st.session_state.is_processing = True
    
    # Check if vector database already exists
    db_path = "./chroma_db"
    db_exists = os.path.exists(db_path) and os.path.isdir(db_path) and os.listdir(db_path)
    
    if not db_exists:
        # Only clear the vector database if it doesn't exist
        clear_vector_db()
        st.info("Creating new vector database...")
    else:
        st.info("Using existing vector database...")
    
    # Process each URL
    all_chunks = []
    processed_urls = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load initial URLs from file
    initial_urls = load_urls_from_file()
    all_urls = set(initial_urls)
    
    # Process each initial URL and collect internal links
    for i, url in enumerate(initial_urls):
        if url not in st.session_state.processed_urls:
            status_text.text(f"Processing URL {i+1}/{len(initial_urls)}: {url}")
            
            # Get internal links
            internal_links = get_internal_links(url, url)
            all_urls.update(internal_links)
            
            # Process the main URL
            content = fetch_url_content(url)
            if content:
                chunks = process_text(content, url)
                all_chunks.append(chunks)
                processed_urls.append(url)
                st.session_state.processed_urls.append(url)
            
            progress_bar.progress((i + 1) / len(initial_urls))
    
    # Process internal links
    internal_urls = list(all_urls - set(initial_urls))
    for i, url in enumerate(internal_urls):
        if url not in st.session_state.processed_urls:
            status_text.text(f"Processing internal URL {i+1}/{len(internal_urls)}: {url}")
            
            content = fetch_url_content(url)
            if content:
                chunks = process_text(content, url)
                all_chunks.append(chunks)
                processed_urls.append(url)
                st.session_state.processed_urls.append(url)
            
            progress_bar.progress((i + 1) / len(internal_urls))
    
    if all_chunks:
        status_text.text("Creating vector store...")
        vectorstore = create_vector_store(all_chunks)
        
        status_text.text("Creating QA chain...")
        st.session_state.qa_chain = create_qa_chain(vectorstore)
        
        status_text.text("Done! You can now ask questions about the content.")
        st.success(f"Successfully processed {len(processed_urls)} URLs!")
    else:
        if db_exists:
            # If we have an existing database but no new content, try to use it
            try:
                status_text.text("Using existing vector store...")
                client = chromadb.PersistentClient(path=db_path)
                vectorstore = Chroma(
                    client=client,
                    collection_name="document_collection",
                    embedding_function=OllamaEmbeddings(model="llama3.2")
                )
                
                status_text.text("Creating QA chain...")
                st.session_state.qa_chain = create_qa_chain(vectorstore)
                
                status_text.text("Done! You can now ask questions about the content.")
                st.success("Using existing vector database. You can now ask questions about the content.")
            except Exception as e:
                st.error(f"Failed to use existing vector database: {str(e)}")
                st.error("No content was extracted from the URLs.")
        else:
            st.error("No content was extracted from the URLs.")
    
    st.session_state.is_processing = False
    st.session_state.initialization_complete = True

# Main UI
st.title("🤖 Document RAG with Llama 3.2")
st.markdown("""
    This application allows you to:
    1. Automatically process URLs from urls.txt file
    2. Crawl internal links from each URL
    3. Ask questions about the content
    4. Get AI-generated responses using Llama 3.2
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
                    st.markdown(f"""
                        <div class="chat-message {message['role']}">
                            <div>{message['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Create a form for the input to ensure it's cleared after submission
        with st.form(key="chat_form", clear_on_submit=True):
            question = st.text_input("Ask a question about the documents:", key="question")
            submit_button = st.form_submit_button("Send")
        
        # Process the question if the form was submitted
        if submit_button and question:
            with st.spinner("Thinking..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Get response
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Update last question
                st.session_state.last_question = question
                
                # Rerun once to update the UI
                st.experimental_rerun()

else:
    if st.session_state.is_processing:
        st.info("Processing URLs and creating the RAG system. Please wait...")
    else:
        st.error("Failed to initialize the RAG system. Please check the logs for errors.") 