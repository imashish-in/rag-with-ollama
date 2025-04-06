import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import os
import time
from urllib.parse import urljoin, urlparse
import shutil
import subprocess
import numpy as np
from functools import lru_cache

# Define the model name as a constant to ensure consistency
MODEL_NAME = "llama3.2"

class VectorDBManager:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.processed_urls = []
        self.all_chunks = []
        self.embeddings = OllamaEmbeddings(model=MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Reduced chunk size for better granularity
            chunk_overlap=200,
            length_function=len,
        )
        
    def clear_vector_db(self):
        """Clear the vector database directory"""
        # First, try to close any open ChromaDB connections
        try:
            # Force garbage collection to close connections
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning during garbage collection: {str(e)}")
        
        # Add a small delay to ensure files are properly closed
        time.sleep(1)
        
        # Now try to remove the directory
        if os.path.exists(self.db_path):
            try:
                # Use a more direct approach to remove the directory
                # On macOS/Linux, use rm -rf
                if os.name != 'nt':  # Not Windows
                    subprocess.run(['rm', '-rf', self.db_path], check=True)
                else:
                    # On Windows, use rmdir /s /q
                    subprocess.run(['rmdir', '/s', '/q', self.db_path], check=True)
                    
                print("Vector database cleared successfully using system command.")
            except Exception as e:
                print(f"Could not remove directory using system command: {str(e)}")
                
                # Fallback to Python's os.remove for individual files
                try:
                    for root, dirs, files in os.walk(self.db_path, topdown=False):
                        for name in files:
                            try:
                                file_path = os.path.join(root, name)
                                os.remove(file_path)
                            except Exception as e:
                                print(f"Could not remove file {file_path}: {str(e)}")
                        
                        for name in dirs:
                            try:
                                dir_path = os.path.join(root, name)
                                os.rmdir(dir_path)
                            except Exception as e:
                                print(f"Could not remove directory {dir_path}: {str(e)}")
                    
                    # Finally try to remove the main directory
                    os.rmdir(self.db_path)
                    print("Vector database cleared successfully using Python's os.remove.")
                except Exception as e:
                    print(f"Could not completely remove directory {self.db_path}: {str(e)}")
        
        # Create a fresh directory
        os.makedirs(self.db_path, exist_ok=True)
        
    @lru_cache(maxsize=100)
    def fetch_url_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Skipping URL {url} - Status code: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find the main content area
            main_content = None
            
            # Look for common content containers
            for selector in ['main', 'article', '.content', '#content', '.main', '#main', '.post', '.entry']:
                content = soup.select_one(selector)
                if content:
                    main_content = content
                    break
            
            # If no main content found, use the body
            if not main_content:
                main_content = soup.body
            
            # Extract text from the main content
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Print the length of the text to verify content was extracted
            print(f"Extracted {len(text)} characters from {url}")
            
            return text
        except Exception as e:
            print(f"Error fetching URL: {str(e)}")
            return None

    def process_text(self, text, url):
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add URL metadata and more context to each chunk
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            # Add more context to help with retrieval
            enhanced_chunk = f"Source: {url}\n\nContent: {chunk}\n\n"
            
            # Add a summary of the chunk if it's long enough
            if len(chunk) > 100:
                summary = f"This text discusses: {chunk[:100]}..."
                enhanced_chunk += f"Summary: {summary}\n\n"
            
            chunks_with_metadata.append(enhanced_chunk)
        
        # Print the number of chunks created
        print(f"Created {len(chunks_with_metadata)} chunks from {url}")
        
        return chunks_with_metadata

    def process_url(self, url, total_urls, current_index):
        """Process a single URL without fetching internal links"""
        try:
            # Check if URL has already been processed
            if url in self.processed_urls:
                return []
            
            # Update status
            print(f"Processing URL {current_index}/{total_urls}: {url}")
            
            # Process the URL
            content = self.fetch_url_content(url)
            chunks = []
            processed_urls = []
            
            if content:
                chunks.extend(self.process_text(content, url))
                processed_urls.append(url)
                
                # Mark URL as processed
                self.processed_urls.append(url)
                
                # Add chunks to the list
                self.all_chunks.extend(chunks)
            
            return processed_urls
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return []

    def create_vector_store(self, chunks_list):
        # Create embeddings and vector store
        embeddings = self.embeddings
        
        # Ensure the database directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB with persistent client
        client = chromadb.PersistentClient(path=self.db_path)
        
        # Print the number of chunks being embedded
        print(f"Creating embeddings for {len(chunks_list)} chunks...")
        
        # Create the vector store
        vectorstore = Chroma.from_texts(
            texts=chunks_list,
            embedding=embeddings,
            client=client,
            collection_name="document_collection"
        )
        
        # Verify the vector store was created correctly
        try:
            # Try a simple similarity search to verify the vector store
            results = vectorstore.similarity_search("test", k=1)
            print(f"Vector store verification successful. Found {len(results)} results.")
        except Exception as e:
            print(f"Warning: Vector store verification failed: {str(e)}")
        
        return vectorstore

    def load_urls_from_file(self, file_path="urls.txt"):
        try:
            with open(file_path, "r") as file:
                urls = [line.strip() for line in file.readlines() if line.strip()]
            
            # Print the number of URLs loaded
            print(f"Loaded {len(urls)} URLs from {file_path}")
            
            return urls
        except Exception as e:
            print(f"Error reading URLs file: {str(e)}")
            return []

    def update_vector_db(self, force_recreate=False):
        """Update the vector database with content from URLs"""
        # Check if vector database already exists
        db_exists = os.path.exists(self.db_path) and os.path.isdir(self.db_path) and os.listdir(self.db_path)
        
        if not db_exists or force_recreate:
            # Clear the vector database if it doesn't exist or if force_recreate is True
            self.clear_vector_db()
            print("Creating new vector database...")
        else:
            print("Using existing vector database...")
        
        # Reset the chunks list
        self.all_chunks = []
        
        # Load initial URLs from file
        initial_urls = self.load_urls_from_file()
        
        if not initial_urls:
            print("No URLs found in urls.txt file.")
            return None
        
        # Process URLs sequentially
        processed_urls = []
        for i, url in enumerate(initial_urls):
            urls = self.process_url(url, len(initial_urls), i+1)
            processed_urls.extend(urls)
        
        if self.all_chunks:
            print(f"Creating vector store with {len(self.all_chunks)} chunks...")
            vectorstore = self.create_vector_store(self.all_chunks)
            print(f"Successfully processed {len(processed_urls)} URLs!")
            
            # Verify the vector store
            try:
                # Try a simple similarity search to verify the vector store
                results = vectorstore.similarity_search("test", k=1)
                print(f"Vector store verification successful. Found {len(results)} results.")
            except Exception as e:
                print(f"Warning: Vector store verification failed: {str(e)}")
            
            return vectorstore
        else:
            if db_exists:
                # If we have an existing database but no new content, try to use it
                try:
                    print("Using existing vector store...")
                    client = chromadb.PersistentClient(path=self.db_path)
                    vectorstore = Chroma(
                        client=client,
                        collection_name="document_collection",
                        embedding_function=self.embeddings
                    )
                    print("Using existing vector database.")
                    
                    # Verify the vector store
                    try:
                        # Try a simple similarity search to verify the vector store
                        results = vectorstore.similarity_search("test", k=1)
                        print(f"Vector store verification successful. Found {len(results)} results.")
                    except Exception as e:
                        print(f"Warning: Vector store verification failed: {str(e)}")
                    
                    return vectorstore
                except Exception as e:
                    print(f"Failed to use existing vector database: {str(e)}")
                    print("No content was extracted from the URLs.")
            else:
                print("No content was extracted from the URLs.")
            return None

def main():
    """Main function to run the vector database manager independently"""
    print("Starting Vector Database Manager...")
    print("This will process URLs from urls.txt and create/update the vector database.")
    
    # Create VectorDBManager instance
    db_manager = VectorDBManager()
    
    # Update the vector database
    print("\nUpdating vector database...")
    vectorstore = db_manager.update_vector_db(force_recreate=True)  # Set to True to force recreation
    
    if vectorstore:
        print("\nVector database updated successfully!")
        print("You can now use the chat interface in app.py to ask questions.")
    else:
        print("\nFailed to update vector database.")
        print("Please check the logs for errors.")

if __name__ == "__main__":
    main() 