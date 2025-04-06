import subprocess
import json
import sys
import time

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        # Check if Ollama is installed
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama is not installed or not in PATH.")
            print(f"Error: {result.stderr}")
            return False
        
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Ollama is not installed or not in PATH.")
        print("Please install Ollama from https://ollama.ai/")
        return False

def check_model_availability(model_name):
    """Check if the specified model is available in Ollama"""
    try:
        # List available models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to list Ollama models: {result.stderr}")
            return False
        
        # Parse the output to check if the model is available
        models = result.stdout.strip().split('\n')
        for model in models:
            if model_name in model:
                print(f"✅ Model '{model_name}' is available in Ollama.")
                return True
        
        print(f"❌ Model '{model_name}' is not available in Ollama.")
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        
        # Ask if the user wants to pull the model
        response = input(f"Do you want to pull the '{model_name}' model? (y/n): ")
        if response.lower() == 'y':
            print(f"Pulling '{model_name}' model...")
            pull_result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
            if pull_result.returncode != 0:
                print(f"❌ Failed to pull '{model_name}' model: {pull_result.stderr}")
                return False
            
            print(f"✅ Successfully pulled '{model_name}' model.")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error checking model availability: {str(e)}")
        return False

def test_embeddings(model_name):
    """Test if the model can generate embeddings"""
    try:
        print(f"Testing embeddings with '{model_name}' model...")
        
        # Create a simple Python script to test embeddings
        test_script = f"""
import time
from langchain_community.embeddings import OllamaEmbeddings

# Initialize embeddings
embeddings = OllamaEmbeddings(model="{model_name}")

# Test embedding generation
start_time = time.time()
result = embeddings.embed_query("This is a test query.")
end_time = time.time()

# Print results
print(f"Embedding dimension: {{len(result)}}")
print(f"Time taken: {{end_time - start_time:.2f}} seconds")
print("Embedding sample: {{result[:5]}}")
"""
        
        # Write the script to a file
        with open("test_embeddings.py", "w") as f:
            f.write(test_script)
        
        # Run the script
        result = subprocess.run([sys.executable, "test_embeddings.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to test embeddings: {result.stderr}")
            return False
        
        print("✅ Embeddings test successful:")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"❌ Error testing embeddings: {str(e)}")
        return False

def main():
    """Main function to verify Ollama installation and model availability"""
    print("🔍 Verifying Ollama installation and model availability...")
    
    # Check if Ollama is installed
    if not check_ollama_installation():
        return
    
    # Check if the model is available
    model_name = "llama3.2"  # Use the same model name as in vector_db_manager.py
    if not check_model_availability(model_name):
        return
    
    # Test embeddings
    if not test_embeddings(model_name):
        return
    
    print("\n✅ All checks passed! Ollama is properly installed and the model is available.")
    print("You can now run vector_db_manager.py to create the vector database and app.py to use the chat interface.")

if __name__ == "__main__":
    main() 