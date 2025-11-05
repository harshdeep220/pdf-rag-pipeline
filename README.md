## 🧠 Ollama-Pinecone-RAG

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that indexes PDF documents, stores their embeddings in **Pinecone**, and answers user queries using **Ollama’s local LLMs (Gemma)**.

---

### 🚀 Features

* 📄 Load and split PDFs using **LangChain**
* 🔍 Generate embeddings with **Ollama EmbeddingGemma**
* 🧩 Store and query embeddings via **Pinecone Vector Database**
* 💬 Generate context-aware answers using **Gemma-3 4B**
* ⚙️ Simple, local-first, and easily extensible

---

### 🛠️ Installation

```bash
# Clone this repo
git clone https://github.com/<your-username>/ollama-pinecone-rag.git
cd ollama-pinecone-rag

# Install dependencies
pip install -r requirements.txt
```

---

### ⚙️ Configuration

Edit the following variables in `main.py` before running:

```python
PDF_PATH = "path/to/your/pdf/file.pdf"
INDEX_NAME = "your-pinecone-index-name"
PINECONE_API_KEY = "your-pinecone-api-key"
```

For security, you can store your API key in an environment variable:

```bash
export PINECONE_API_KEY="your-pinecone-api-key"
```

---

### ▶️ Usage

```bash
python main.py
```

This will:

1. Load your PDF
2. Chunk it into text segments
3. Generate and upload embeddings to Pinecone
4. Answer your test query:

   > "Summarize the main findings of the PDF."

---

### 📦 Project Structure

```
ollama-pinecone-rag/
│
├── main.py             # Main RAG pipeline script
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

### 🧩 Next Steps

* Add a Flask or Streamlit UI for interactive querying
* Integrate multi-document retrieval
* Add reranking or summarization before context generation

---
