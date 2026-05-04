streamlit : https://captone-3-chatbot-film.streamlit.app/

# 🎬 Movies Recommendation base on Feeling

> Chatbot rekomendasi film berbasis mood menggunakan RAG (Retrieval-Augmented Generation) dengan database IMDB Top 1000 dan Qdrant Vector Database.

---

## 📌 Deskripsi Project

**Movies Recommendation base on Feeling** adalah aplikasi chatbot cerdas yang merekomendasikan film berdasarkan mood/perasaan pengguna. Pengguna memilih mood mereka, lalu sistem secara otomatis mencari film yang paling relevan menggunakan semantic search dari vector database Qdrant, kemudian LLM (GPT-4o-mini) menghasilkan rekomendasi yang personal dan deskriptif dalam Bahasa Indonesia.

---

## 🏗️ Arsitektur Sistem

```
User → Pilih Mood
           ↓
    Mood → Genre Mapping
           ↓
    OpenAI Embedding (text-embedding-3-small)
           ↓
    Qdrant Semantic Search (Top 5 Film)
           ↓
    ReAct Agent (GPT-4o-mini)
           ↓
    Rekomendasi Film (Bahasa Indonesia)
           ↓
    Streamlit UI
```

---

## 🎭 Fitur Utama

- **Mood-Based Recommendation** — 6 pilihan mood yang masing-masing terhubung ke genre film yang relevan
- **Semantic Search** — menggunakan Qdrant Vector Database untuk mencari film berdasarkan makna, bukan hanya keyword
- **Dynamic Background** — tampilan UI berubah sesuai mood yang dipilih
- **RAG Agent** — ReAct Agent dengan LangGraph yang otomatis menggunakan tools untuk mengambil data film
- **Token & Cost Tracking** — menampilkan estimasi biaya penggunaan API per sesi
- **Chat History** — menyimpan riwayat percakapan dalam sesi

---

## 🎨 Mood & Genre Mapping

| Mood | Genre | Background |
|------|-------|------------|
| 😊 Senang | Comedy, Animation, Adventure, Family | 🌈 Kuning-Orange |
| 😢 Sedih | Drama, Romance, War | 🌧️ Biru Gelap |
| 🤔 Berpikir | Biography, History, Mystery | 💭 Abu-Biru |
| 😱 Tegang | Crime, Thriller, Horror | 👻 Merah-Hitam |
| 🚀 Semangat | Action, Adventure, Sci-Fi | 🚀 Ungu-Hitam |
| ❤️ Romantis | Romance, Comedy, Drama | ❤️ Merah-Pink |

---

## 🗂️ Struktur Project

```
Capstone 3/
├── .streamlit/
│   └── secrets.toml          # Credentials (tidak di-commit)
├── chatbot/
│   └── data/
│       └── raw/
│           └── imdb_top_1000.csv   # Dataset utama
├── capstone_3_env/           # Virtual environment
├── .env                      # Environment variables (tidak di-commit)
├── .gitignore
├── db.py                     # Script ingest data ke Qdrant
├── main.py                   # Aplikasi Streamlit utama
├── cek_qdrant.py             # Script verifikasi data Qdrant
└── README.md
```

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| **Frontend** | Streamlit |
| **LLM** | GPT-4o-mini (OpenAI) |
| **Embedding** | text-embedding-3-small (OpenAI) |
| **Vector DB** | Qdrant Cloud |
| **RAG Framework** | LangChain + LangGraph |
| **Agent** | ReAct Agent (LangGraph) |
| **Dataset** | IMDB Top 1000 Movies |
| **Language** | Python 3.13 |

---

## ⚙️ Instalasi & Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/capstone-3.git
cd capstone-3
```

### 2. Buat Virtual Environment
```bash
python -m venv capstone_3_env
.\capstone_3_env\Scripts\activate  # Windows
source capstone_3_env/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install streamlit langchain langchain-openai langchain-qdrant langgraph qdrant-client pandas python-dotenv uuid
```

### 4. Setup Credentials

Buat file `.env`:
```env
QDRANT_URL="https://your-cluster.qdrant.io"
QDRANT_API_KEY="your-qdrant-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

Buat file `.streamlit/secrets.toml`:
```toml
QDRANT_URL="https://your-cluster.qdrant.io"
QDRANT_API_KEY="your-qdrant-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### 5. Ingest Data ke Qdrant
```bash
python db.py
```

### 6. Verifikasi Data
```bash
python cek_qdrant.py
# Output: Total points: 1000 | Collection status: green
```

### 7. Jalankan Aplikasi
```bash
streamlit run main.py
```

---

## 🔄 Alur Kerja RAG

### Ingestion (db.py)
```
CSV 1000 Movies
      ↓
Pandas DataFrame (cleaning)
      ↓
Buat Document (Series_Title + Overview sebagai content)
      ↓
OpenAI Embedding → Vector 1536 dimensi
      ↓
Upload ke Qdrant collection "Data_IMDB" (batch 50)
```

### Retrieval & Generation (main.py)
```
User pilih Mood
      ↓
Mood → mapped ke Genre query
      ↓
Query di-embed → similarity search k=5
      ↓
5 film relevan dikirim ke ReAct Agent
      ↓
GPT-4o-mini generate rekomendasi (Bahasa Indonesia)
```

---

## 📊 Dataset

- **Sumber**: IMDB Top 1000 Movies
- **Jumlah Data**: 1.000 film
- **Kolom Utama**: `Series_Title`, `Overview`, `Genre`, `IMDB_Rating`, `Director`, `Released_Year`, `Gross`
- **Kolom yang digunakan untuk embedding**: `Series_Title` + `Overview`
- **Metadata yang disimpan**: `film_id`, `Series_Title`

---

## 💰 Estimasi Biaya API

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o-mini | $0.15 / 1M tokens | $0.60 / 1M tokens |
| text-embedding-3-small | $0.02 / 1M tokens | - |

*Estimasi biaya ditampilkan langsung di UI per sesi chat dalam Rupiah (kurs Rp 17.000/USD)*

---

## 🚀 Demo

1. Buka aplikasi → pilih mood kamu
2. Chatbot otomatis merekomendasikan 3-5 film sesuai mood
3. Tanya lebih lanjut tentang film tertentu
4. Klik "Ganti Mood" untuk rekomendasi genre berbeda

---

## ⚠️ Catatan

- Pastikan Qdrant cluster dalam status **HEALTHY** sebelum menjalankan aplikasi
- File `.env` dan `secrets.toml` **jangan di-commit** ke GitHub (sudah ada di `.gitignore`)
- Jalankan `db.py` hanya sekali — script sudah otomatis hapus collection lama sebelum ingest ulang

---

## 👨‍💻 Author

**Muda Zhuhri** — AI Engineering Student, Purwadhika Digital Technology School  
Module 3 Capstone Project
