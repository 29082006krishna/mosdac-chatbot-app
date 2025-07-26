import json
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

MOSDAC_BASE_URL = "https://www.mosdac.gov.in"
VECTOR_DB_PATH = "./mosdac_vector_db"
JSON_FILE_PATH = "mosdac_data.json"

def load_json_data(file_path):
    """Loads JSON data from a specified file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def scrape_url_content(url):
    """Fetches content from a URL and extracts meaningful text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find common content areas
        # This is a basic approach. For more complex sites, you might need more specific selectors.
        content_div = soup.find('div', class_='content-wrapper') or \
                      soup.find('main') or \
                      soup.find('article') or \
                      soup.find('body')

        if content_div:
            # Extract text, remove script/style tags
            for script_or_style in content_div(["script", "style"]):
                script_or_style.extract()
            text = content_div.get_text(separator=' ', strip=True)
            return text
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while scraping {url}: {e}")
        return ""

def process_menu_item(item, parent_path="", depth=0):
    """Recursively processes menu items to create text chunks."""
    name = item.get("Menu") or item.get("Name") or "" # Ensure name is always a string
    relative_url = item.get("URL")
    full_url = f"{MOSDAC_BASE_URL}{relative_url}" if relative_url and not str(relative_url).startswith("#") else None
# Added str() cast just in case relative_url itself could somehow not be a string.

    text_chunk = ""
    metadata = {
        "name": name,
        "url": full_url,
        "path": (parent_path + " > " + name) if parent_path else name,
        "depth": depth
    }

    if depth == 0: # Main menu items
        text_chunk = f"The main MOSDAC menu item is '{name}'. "
        if full_url:
            text_chunk += f"It can be accessed at {full_url}. "
        if item.get("Sub_Menu"):
            sub_menus = ", ".join([s.get("Name") for s in item["Sub_Menu"] if s.get("Name")])
            text_chunk += f"It contains sub-sections like: {sub_menus}."
    elif depth == 1: # First level sub-menus
        text_chunk = f"Under '{parent_path}', there is a sub-menu item: '{name}'. "
        if full_url:
            text_chunk += f"Its direct URL is {full_url}. "
        if item.get("Sub_Sub_Menu"):
            sub_sub_categories = ", ".join([s.get("Category") for s in item["Sub_Sub_Menu"] if s.get("Category")])
            if sub_sub_categories:
                text_chunk += f"It further categorizes data into: {sub_sub_categories}."
    elif depth == 2: # Second level sub-menus (like categories under Open Data)
        if "Category" in item:
            category_name = item["Category"]
            items_list = ", ".join([i.get("Name") for i in item["Items"] if i.get("Name")])
            text_chunk = f"Within the '{parent_path}' section, there is a category called '{category_name}'. It contains data products such as: {items_list}."
            # For categories, the URL is typically for the individual items, not the category itself.
            metadata["url"] = None
        else: # Regular sub-sub-menu item
            text_chunk = f"Under '{parent_path}', there is a nested menu item: '{name}'. "
            if full_url:
                text_chunk += f"You can find more details at {full_url}. "
    elif depth == 3: # Individual items within categories (e.g., specific data products)
        text_chunk = f"A specific MOSDAC data product or section is '{name}'. "
        if full_url:
            text_chunk += f"You can find more details about it at {full_url}. "
        text_chunk += f"This is part of the '{parent_path}' section."


    documents.append(Document(page_content=text_chunk, metadata=metadata))

    # Add scraped content from the URL if it's a valid, non-hash URL
    if full_url and not full_url.endswith("#"):
        scraped_content = scrape_url_content(full_url)
        if scraped_content:
            # Create a new document for the scraped content
            # This allows the LLM to retrieve detailed information from the page
            documents.append(Document(
                page_content=f"Detailed information from the MOSDAC page '{name}' at {full_url}: {scraped_content}",
                metadata={
                    "source": full_url,
                    "name": name,
                    "type": "scraped_page_content",
                    "path": metadata["path"],
                    "depth": depth
                }
            ))

    # Recursively process sub-menus/items
    if "Sub_Menu" in item:
        for sub_item in item["Sub_Menu"]:
            process_menu_item(sub_item, metadata["path"], depth + 1)
    if "Sub_Sub_Menu" in item:
        for sub_sub_item in item["Sub_Sub_Menu"]:
            # If it's a category, process its items
            if "Category" in sub_sub_item and "Items" in sub_sub_item:
                process_menu_item(sub_sub_item, metadata["path"], depth + 2)
            else: # Regular sub-sub-menu item
                process_menu_item(sub_sub_item, metadata["path"], depth + 2)


if __name__ == "__main__":
    print("Starting vector database creation...")

    # Load the JSON menu data
    mosdac_menu_data = load_json_data(JSON_FILE_PATH)
    documents = []

    # Process JSON data to create initial documents
    print("Processing JSON menu data and scraping linked pages...")
    for item in mosdac_menu_data:
        process_menu_item(item)

    print(f"Generated {len(documents)} initial documents from JSON and scraped content.")

    # Split documents into smaller chunks for better RAG performance
    # This is crucial for handling longer scraped pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap to maintain context between chunks
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks after text splitting.")

    # Initialize Google Gemini Embeddings
    # Use the model that is suitable for embeddings (e.g., 'models/text-embedding-004')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Create and persist the Chroma vector database
    print(f"Creating/updating ChromaDB at {VECTOR_DB_PATH}...")
    # Chroma.frssom_documents will create it if it doesn't exist, otherwise add to it.
    # To clear and recreate: delete the mosdac_vector_db folder before running.
    # Corrected lines:
    vectorstore = Chroma.from_documents(
    	documents=split_docs,
    	embedding=embeddings,
    	persist_directory=VECTOR_DB_PATH # <-- Persistence is now handled by this argument
    )
# No need for vectorstore.persist()
    print("Vector database created and persisted successfully!")
    print(f"Total documents in vector store: {vectorstore._collection.count()}")