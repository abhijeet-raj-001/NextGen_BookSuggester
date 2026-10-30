# indexer.py
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store(csv_path="data/enriched_books.csv", persist_dir="./chroma_books_db"):
    df = pd.read_csv(csv_path).fillna("Unknown")
    documents = []

    for _, row in df.iterrows():
        # Text block representing full semantic context for the embedding model
        page_content = (
            f"Title: {row['Book-Title']}\n"
            f"Author: {row['Book-Author']}\n"
            f"Year: {row['Year-Of-Publication']}\n"
            f"Publisher: {row['Publisher']}\n"
            f"Genres/Categories: {row['Categories']}\n"
            f"Overview: {row['Description']}"
        )
        
        metadata = {
            "title": str(row["Book-Title"]),
            "author": str(row["Book-Author"]),
            "year": str(row["Year-Of-Publication"]),
            "image": str(row["Image-URL-M"]),
            "categories": str(row["Categories"])
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"Stored {len(documents)} books in {persist_dir}")

if __name__ == "__main__":
    build_vector_store()