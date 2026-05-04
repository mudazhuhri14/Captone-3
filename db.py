import pandas as pd
from uuid import uuid4
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

load_dotenv()

data_path = data_path = r"chatbot\data\raw\imdb_top_1000.csv"  
df = pd.read_csv(data_path)

df=df.replace({'Released_Year': 'PG'}, None)

df['Gross'] = df['Gross'].str.replace(',', '', regex=True)

df[['Released_Year','Gross']] = df[['Released_Year','Gross']].apply(pd.to_numeric)

df['film_id'] = [str(uuid4()) for _ in range(len(df['Series_Title']))]

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embedding = OpenAIEmbeddings(
        model='text-embedding-3-small',
    )
url = os.getenv("QDRANT_URL")
qdrant_api = os.getenv("QDRANT_API_KEY")

documents = []

for i in range(len(df)):
    judul_film = df['Series_Title'][i]
    overview_film = df['Overview'][i]
    id_film = df['film_id'][i]
    input_rag = f"Series_Title: {judul_film}, Overview: {overview_film}"
    doc = Document(
        page_content=input_rag,
        metadata={
            "film_id": id_film,
            "Series_Title": judul_film
        },
    )
    documents.append(doc)

uuids = [str(uuid4()) for _ in range(len(documents))]

from qdrant_client.models import Distance, VectorParams

if client.collection_exists("Data_IMDB"):
    client.delete_collection("Data_IMDB")
    print("🗑️ Collection lama dihapus")

client.create_collection(
    collection_name="Data_IMDB",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print("✅ Collection baru dibuat")

qdrant = QdrantVectorStore.from_documents(
    documents[:5],          # upload 5 dulu untuk inisialisasi collection
    embedding=embedding,
    url=url,
    api_key=qdrant_api,
    collection_name="Data_IMDB",
    timeout=60,
)

# Upload sisanya per batch
BATCH_SIZE = 50
total = len(documents)

for i in range(5, total, BATCH_SIZE):
    batch = documents[i : i + BATCH_SIZE]
    qdrant.add_documents(batch)
    print(f"✅ Uploaded {min(i + BATCH_SIZE, total)}/{total} documents")

print("🎉 Create qdrant data success!")

# Cek jumlah data di Qdrant
collection_info = client.get_collection("Data_IMDB")
print(f"Total vectors di Qdrant: {collection_info.vectors_count}")