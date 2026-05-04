import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# ── Load environment variables ──────────────────────────
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Load CSV ─────────────────────────────────────────────
df = pd.read_csv("imdb_top_1000.csv")

# Drop rows with missing critical fields
df = df.dropna(subset=["Series_Title", "Overview", "Genre", "Director"])

print(f"✅ Total movies loaded: {len(df)}")

# ── Convert rows to LangChain Documents ─────────────────
documents = []

for _, row in df.iterrows():
    # Gabungkan semua info jadi satu teks untuk di-embed
    content = f"""
Title: {row['Series_Title']}
Year: {row['Released_Year']}
Genre: {row['Genre']}
Rating: {row['IMDB_Rating']}
Director: {row['Director']}
Stars: {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}
Runtime: {row['Runtime']}
Overview: {row['Overview']}
""".strip()

    # Metadata untuk filtering (opsional)
    metadata = {
        "title": row["Series_Title"],
        "year": int(row["Released_Year"]),
        "genre": row["Genre"],
        "rating": float(row["IMDB_Rating"]),
        "director": row["Director"],
        "stars": f"{row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}",
        "runtime": row["Runtime"],
    }

    documents.append(Document(page_content=content, metadata=metadata))

print(f"✅ Total documents created: {len(documents)}")

# ── Setup Embeddings ─────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# ── Upload ke Qdrant ─────────────────────────────────────
print("⏳ Uploading to Qdrant... (this may take a moment)")

collection_name = "imdb_movies"

qdrant = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
)

print(f"🎉 Done! {len(documents)} movies uploaded to Qdrant collection: '{collection_name}'")
