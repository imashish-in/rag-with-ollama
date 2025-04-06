# RAG with Ollama

A Retrieval-Augmented Generation (RAG) system built with Ollama, LangChain, and Streamlit.

## Overview

This project implements a RAG system that:
1. Fetches content from specified URLs
2. Creates embeddings using Ollama
3. Stores the embeddings in a vector database (ChromaDB)
4. Provides a chat interface for querying the information

## Features

- URL content extraction and processing
- Vector database creation and management
- Interactive chat interface with Streamlit
- Efficient content retrieval using semantic search

## Requirements

- Python 3.11
- Conda (for environment management)
- Ollama (with llama3.2 model)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-with-ollama.git
   cd rag-with-ollama
   ```

2. Create and activate a conda environment with Python 3.11:
   ```bash
   # Create conda environment
   conda create -p rag-ollama python=3.11
   
   # Activate the environment
   conda activate rag-ollama/
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Ollama is installed and the llama3.2 model is available:
   ```bash
   ollama pull llama3.2
   ```

## Usage

1. Add URLs to the `urls.txt` file:
   ```
   https://example.com/page1
   https://example.com/page2
   ```

2. Run the vector database manager to process URLs and create embeddings:
   ```bash
   python vector_db_manager.py
   ```

3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Use the chat interface to ask questions about the content from the URLs.

## Project Structure

- `vector_db_manager.py`: Handles URL processing and vector database creation
- `app.py`: Streamlit application for the chat interface
- `urls.txt`: List of URLs to process
- `chroma_db/`: Directory for the vector database (created automatically)

