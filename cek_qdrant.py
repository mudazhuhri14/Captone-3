from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

info = client.get_collection("Data_IMDB")
print(f"Total points: {info.points_count}")
print(f"Collection status: {info.status}")