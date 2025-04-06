import os
import argparse

def add_urls_to_file(urls=None, file_path="urls.txt"):
    """Add URLs to the urls.txt file"""
    # Check if the file exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
    
    # Read existing URLs
    with open(file_path, "r") as f:
        existing_urls = [line.strip() for line in f.readlines() if line.strip()]
    
    # Print existing URLs
    print(f"Existing URLs ({len(existing_urls)}):")
    for url in existing_urls:
        print(f"  - {url}")
    
    # If no URLs provided, ask for input
    if not urls:
        print("\nEnter URLs (one per line, press Enter twice to finish):")
        urls = []
        while True:
            url = input().strip()
            if not url:
                break
            if url.startswith(('http://', 'https://')):
                urls.append(url)
            else:
                print("Warning: URL must start with http:// or https://")
    
    # Filter out URLs that already exist
    new_urls = [url for url in urls if url not in existing_urls]
    
    # Add new URLs to the file
    if new_urls:
        with open(file_path, "a") as f:
            for url in new_urls:
                f.write(f"{url}\n")
        
        print(f"\nAdded {len(new_urls)} new URLs to {file_path}:")
        for url in new_urls:
            print(f"  - {url}")
    else:
        print("\nNo new URLs to add.")
    
    # Print total URLs
    with open(file_path, "r") as f:
        all_urls = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"\nTotal URLs in {file_path}: {len(all_urls)}")

def main():
    """Main function to add URLs to the urls.txt file"""
    parser = argparse.ArgumentParser(description="Add URLs to urls.txt file")
    parser.add_argument("--urls", type=str, help="Comma-separated list of URLs to add")
    parser.add_argument("--file", default="urls.txt", help="Path to the URLs file (default: urls.txt)")
    args = parser.parse_args()

    # Process URLs if provided
    urls = None
    if args.urls:
        urls = [url.strip() for url in args.urls.split(',') if url.strip()]

    print("🔗 Adding URLs to urls.txt...")
    add_urls_to_file(urls, args.file)
    print("\n✅ URLs added successfully!")
    print("You can now run vector_db_manager.py to create the vector database with the updated URLs.")

if __name__ == "__main__":
    main() 