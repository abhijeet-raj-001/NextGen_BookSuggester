# enrich_data.py
import requests
import pandas as pd
import time

def fetch_book_metadata(isbn, title, author):
    """Fetches plot description and categories from Google Books API."""
    query = f"isbn:{isbn}" if pd.notna(isbn) else f"intitle:{title}+inauthor:{author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(query)}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            items = response.json().get("items", [])
            if items:
                volume = items[0].get("volumeInfo", {})
                return {
                    "description": volume.get("description", "No description available."),
                    "categories": ", ".join(volume.get("categories", ["General"])),
                    "page_count": volume.get("pageCount", "Unknown")
                }
    except Exception:
        pass
    return {"description": "No description available.", "categories": "General", "page_count": "Unknown"}

def build_enriched_dataset(books_path="Books.csv", output_path="enriched_books.csv", sample_limit=500):
    books = pd.read_csv(books_path, low_memory=False).drop_duplicates(subset=["Book-Title"])
    candidates = books.head(sample_limit).copy()

    descriptions, categories, pages = [], [], []
    for _, row in candidates.iterrows():
        meta = fetch_book_metadata(row["ISBN"], row["Book-Title"], row["Book-Author"])
        descriptions.append(meta["description"])
        categories.append(meta["categories"])
        pages.append(meta["page_count"])
        time.sleep(0.1)  # Respect API rate limits

    candidates["Description"] = descriptions
    candidates["Categories"] = categories
    candidates["PageCount"] = pages
    candidates.to_csv(output_path, index=False)
    print(f"Enriched {len(candidates)} records saved to {output_path}")

if __name__ == "__main__":
    build_enriched_dataset()