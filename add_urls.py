import os

def add_urls_to_file(file_path="urls.txt"):
    """Add more URLs to the urls.txt file"""
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
    
    # Additional URLs to add
    additional_urls = [
        "https://lakshmikumari.in/about/",
        "https://lakshmikumari.in/contact/",
        "https://lakshmikumari.in/blog/",
        "https://lakshmikumari.in/services/",
        "https://lakshmikumari.in/portfolio/",
        "https://lakshmikumari.in/testimonials/",
        "https://lakshmikumari.in/faq/",
        "https://lakshmikumari.in/privacy-policy/",
        "https://lakshmikumari.in/terms-of-service/",
        "https://lakshmikumari.in/sitemap/",
    ]
    
    # Filter out URLs that already exist
    new_urls = [url for url in additional_urls if url not in existing_urls]
    
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
    print("🔗 Adding URLs to urls.txt...")
    add_urls_to_file()
    print("\n✅ URLs added successfully!")
    print("You can now run vector_db_manager.py to create the vector database with the updated URLs.")

if __name__ == "__main__":
    main() 