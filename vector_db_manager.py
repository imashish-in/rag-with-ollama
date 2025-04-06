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
import re
from typing import List, Optional

# Define the model name as a constant to ensure consistency
MODEL_NAME = "llama3.2"  # Using llama3.2 model

class VectorDBManager:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.processed_urls = []
        self.all_chunks = []
        self.embeddings = OllamaEmbeddings(
            model=MODEL_NAME,
            temperature=0.0,  # Lower temperature for more consistent embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for more granular chunks
            chunk_overlap=100,  # Reduced overlap but still maintaining context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # More granular separators
        )
        
    def clear_vector_db(self):
        """Clear the vector database directory"""
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print("Vector database cleared successfully.")
            os.makedirs(self.db_path, exist_ok=True)
        except Exception as e:
            print(f"Error clearing vector database: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove multiple punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        return text.strip()

    def extract_text_from_element(self, element) -> str:
        """Extract and clean text from a BeautifulSoup element"""
        if not element:
            return ""
        
        # Remove unwanted elements
        for unwanted in element.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
            unwanted.decompose()
        
        # Get text and clean it
        text = element.get_text(separator=' ', strip=True)
        return self.clean_text(text)

    def fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract content from a URL with improved error handling and content extraction"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find the main content
            content_elements = []
            
            # Look for article content first
            for article in soup.find_all(['article', 'main']):
                content_elements.append(self.extract_text_from_element(article))
            
            # If no article/main content, look for common content containers
            if not content_elements:
                for selector in ['.content', '#content', '.post', '.entry', '.article', '#main-content']:
                    element = soup.select_one(selector)
                    if element:
                        content_elements.append(self.extract_text_from_element(element))
            
            # If still no content, try to extract from body but exclude header and footer
            if not content_elements:
                body = soup.body
                if body:
                    # Remove header and footer if present
                    for unwanted in body.find_all(['header', 'footer', 'nav']):
                        unwanted.decompose()
                    content_elements.append(self.extract_text_from_element(body))
            
            # Combine all extracted content
            text = ' '.join(content_elements)
            
            if not text:
                print(f"Warning: No content extracted from {url}")
                return None
            
            print(f"Successfully extracted {len(text)} characters from {url}")
            return text
            
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error processing {url}: {str(e)}")
            return None

    def process_text(self, text: str, url: str) -> List[str]:
        """Process text into chunks with improved metadata and context"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Add metadata and context to each chunk
            chunks_with_metadata = []
            for i, chunk in enumerate(chunks):
                # Clean the chunk text
                cleaned_chunk = self.clean_text(chunk)
                
                # Skip empty chunks
                if not cleaned_chunk:
                    continue
                
                # Create enhanced chunk with metadata
                enhanced_chunk = f"""Source: {url}
Content: {cleaned_chunk}
Context: Chunk {i+1} of {len(chunks)}
Summary: {cleaned_chunk[:150]}...
"""
                chunks_with_metadata.append(enhanced_chunk)
            
            print(f"Created {len(chunks_with_metadata)} chunks from {url}")
            return chunks_with_metadata
            
        except Exception as e:
            print(f"Error processing text from {url}: {str(e)}")
            return []

    def process_url(self, url: str, total_urls: int, current_index: int) -> List[str]:
        """Process a single URL with improved error handling"""
        try:
            if url in self.processed_urls:
                print(f"Skipping already processed URL: {url}")
                return []
            
            print(f"\nProcessing URL {current_index}/{total_urls}: {url}")
            content = self.fetch_url_content(url)
            
            if not content:
                print(f"No content extracted from {url}")
                return []
            
            chunks = self.process_text(content, url)
            if chunks:
                self.processed_urls.append(url)
                self.all_chunks.extend(chunks)
                return [url]
            
            return []
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return []

    def create_vector_store(self, chunks_list: List[str]):
        """Create vector store with improved error handling and verification"""
        try:
            if not chunks_list:
                print("No chunks to process. Vector store creation failed.")
                return None
            
            print(f"\nCreating vector store with {len(chunks_list)} chunks...")
            
            # Ensure the database directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB with persistent client
            client = chromadb.PersistentClient(path=self.db_path)
            
            # Create the vector store with metadata
            vectorstore = Chroma.from_texts(
                texts=chunks_list,
                embedding=self.embeddings,
                client=client,
                collection_name="document_collection"
            )
            
            # Verify the vector store
            results = vectorstore.similarity_search("test", k=1)
            print(f"Vector store created and verified successfully with {len(chunks_list)} chunks.")
            
            return vectorstore
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None

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

    def update_vector_db(self, force_recreate=True):
        """Update the vector database with improved error handling and progress tracking"""
        try:
            if force_recreate:
                self.clear_vector_db()
                print("\nCreating new vector database...")
            
            # Reset state
            self.all_chunks = []
            self.processed_urls = []
            
            # Load and validate URLs
            urls = self.load_urls_from_file()
            if not urls:
                print("No URLs found in urls.txt file.")
                return None
            
            print(f"\nProcessing {len(urls)} URLs...")
            
            # Process each URL
            processed_urls = []
            for i, url in enumerate(urls, 1):
                processed = self.process_url(url, len(urls), i)
                processed_urls.extend(processed)
            
            # Create vector store if we have chunks
            if self.all_chunks:
                print(f"\nProcessed {len(processed_urls)} URLs successfully.")
                print(f"Total chunks collected: {len(self.all_chunks)}")
                return self.create_vector_store(self.all_chunks)
            else:
                print("\nNo content was extracted from any URL.")
                return None
                
        except Exception as e:
            print(f"Error updating vector database: {str(e)}")
            return None

def main():
    """Main function to run the vector database manager independently"""
    print("Starting Vector Database Manager...")
    print("This will process URLs from urls.txt and create/update the vector database.")
    
    try:
        # Create VectorDBManager instance
        db_manager = VectorDBManager()
        
        # Update the vector database
        print("\nUpdating vector database...")
        vectorstore = db_manager.update_vector_db(force_recreate=True)
        
        if vectorstore:
            print("\n✅ Vector database updated successfully!")
            print("You can now use the chat interface in app.py to ask questions.")
        else:
            print("\n❌ Failed to update vector database.")
            print("Please check the logs above for errors.")
            
    except Exception as e:
        print(f"\n❌ Error running vector database manager: {str(e)}")
        print("Please check the logs above for details.")

if __name__ == "__main__":
    main() 