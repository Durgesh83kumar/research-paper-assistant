# 🔬 Research Paper Assistant

> An AI-powered research paper analysis tool built with RAG (Retrieval-Augmented Generation) architecture.

Upload any PDF research paper to get instant structured summaries and ask questions in a **ChatGPT-style interface** — powered by LangChain, FAISS, and Groq.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)

---

## 📺 Demo
> 🚀 **[Live Demo](https://research-paper-assistant-ngudxfnwknogw8x5skmsis.streamlit.app/)** — Try it now!

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **PDF Upload** | Upload research papers directly in the chat via **+** button |
| 📋 **Auto Summary** | Instant structured summary covering problem, approach, results and contributions |
| 💬 **RAG Q&A** | Ask questions, get answers grounded in paper content with page citations |
| 📊 **LLM Evaluation** | Every answer scored on Relevance, Faithfulness and Completeness |
| 🧠 **Memory** | Remembers full conversation context across the session |
| 🗂️ **Multi-Chat** | Multiple independent chat sessions like ChatGPT |
| 💾 **Persistence** | Chat history survives browser refresh and app restart |
| 🤖 **General Q&A** | Works as a general AI assistant even without a PDF |
| 🐳 **Docker Ready** | Fully containerized and available on Docker Hub |

---

## 🏗️ Architecture

┌─────────────────────────────────────────────┐

│           STREAMLIT CHAT INTERFACE           │

│     Upload PDF (+)  │  Chat  │  Sidebar      │

└──────────────┬──────────────────────────────┘

│

┌───────▼────────┐

│ PDF INGESTION  │

│ PyMuPDF        │

│ Text Splitter  │

└───────┬────────┘

│

┌───────▼────────┐      ┌──────────────────────────┐

│  FAISS INDEX   │      │       GROQ LLM           │

│  (per-chat)    │─────►│  Chat: Llama 3.3 70B     │

│  HuggingFace   │      │  Summary: Llama 3.1 8B   │

│  Embeddings    │      │  Eval: Llama 3.1 8B      │

└────────────────┘      └──────────────────────────┘

### Data Flow

PDF Upload → Extract Text → Split Chunks → Embed → FAISS Store

│

User Question → Embed Question → Similarity Search ──────┘

│

Top-K Chunks

│

Groq LLM Generates Answer

│

LLM Evaluates Answer Quality

│

Display in Chat UI ✅

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.58 | Chat UI with sidebar |
| **LLM (Chat)** | Groq Llama 3.3 70B | Best quality Q&A |
| **LLM (Summary/Eval)** | Groq Llama 3.1 8B | Fast summarization & evaluation |
| **Embeddings** | HuggingFace MiniLM | Text → Vectors |
| **Vector DB** | FAISS | Per-chat semantic similarity search |
| **Orchestration** | LangChain 1.3 | Pipeline management |
| **PDF Parsing** | PyMuPDF | Text extraction |
| **Evaluation** | LLM-as-a-Judge | Answer quality scoring |
| **Storage** | JSON | Persistent chat history |
| **Container** | Docker | Deployment |

---

## 📁 Project Structure

research-paper-assistant/

│

├── app/

│   └── main.py                  # Streamlit frontend (single page)

│

├── src/

│   ├── llm/

│   │   └── factory.py           # LLM factory (Groq/Gemini, use-case aware)

│   ├── embeddings/

│   │   └── embedder.py          # HuggingFace embedding model

│   ├── ingestion/

│   │   ├── pdf_loader.py        # PDF text extraction

│   │   └── chunker.py           # Text splitting

│   ├── vectorstore/

│   │   └── faiss_store.py       # Per-chat FAISS index management

│   ├── chains/

│   │   ├── summarization.py     # Map-Reduce summarization chain

│   │   └── rag_chain.py         # RAG Q&A with conversational memory

│   ├── evaluation/

│   │   └── evaluator.py         # LLM-as-a-Judge evaluation system

│   └── storage/

│       └── chat_store.py        # JSON persistent chat storage

│

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── requirements.docker.txt

├── .env.example

└── README.md

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Groq API key (free) → [console.groq.com](https://console.groq.com)

### 1. Clone the repository

🔗 **Repository:** [github.com/Durgesh83kumar/research-paper-assistant](https://github.com/Durgesh83kumar/research-paper-assistant)

```bash
git clone https://github.com/Durgesh83kumar/research-paper-assistant.git
cd research-paper-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```ini
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_gemini_key_here
LLM_PROVIDER=groq
EMBEDDING_PROVIDER=huggingface
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
```

### 5. Run the app
```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

### Pull from Docker Hub (Easiest)
```bash
docker pull durg1234/research-paper-assistant
docker run -p 8501:8501 --env-file .env durg1234/research-paper-assistant
```

### Using Docker Compose
```bash
docker-compose up --build
```

### Using Docker directly
```bash
docker build -t research-paper-assistant .
docker run -p 8501:8501 --env-file .env research-paper-assistant
```

App available at `http://localhost:8501`

🐳 **Docker Hub:** [hub.docker.com/r/durg1234/research-paper-assistant](https://hub.docker.com/r/durg1234/research-paper-assistant)

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM |
| `GOOGLE_API_KEY` | ⚠️ Optional | Gemini API key (alternative LLM) |
| `LLM_PROVIDER` | ✅ Yes | `groq` or `gemini` |
| `EMBEDDING_PROVIDER` | ✅ Yes | `huggingface` or `gemini` |
| `CHUNK_SIZE` | ✅ Yes | Text chunk size (default: 1000) |
| `CHUNK_OVERLAP` | ✅ Yes | Chunk overlap (default: 200) |
| `RETRIEVAL_K` | ✅ Yes | Number of chunks to retrieve (default: 5) |

---

## 📊 Evaluation System

Every answer is automatically evaluated using **LLM-as-a-Judge** technique — an industry-standard approach used by OpenAI, Google, and Anthropic:

### PDF Q&A Evaluation
| Metric | Description |
|---|---|
| **Relevance** | Does the answer directly address the question? |
| **Faithfulness** | Is the answer grounded in paper content (not hallucinated)? |
| **Completeness** | Does it fully cover the question? |

### General Chat Evaluation
| Metric | Description |
|---|---|
| **Relevance** | Does the answer address the question? |
| **Clarity** | Is the answer clear and easy to understand? |
| **Completeness** | Does it fully cover the question? |

---

## 💡 How to Use

1. **Start the app** and open `http://localhost:8501`
2. **Upload a PDF** by clicking the **+** button in the chat input
3. **Wait** for the paper to be indexed and summarized (30-60 seconds)
4. **Ask questions** about the paper in natural language
5. **View scores** to see how good each answer is
6. **Start a new chat** using the **➕ New Chat** button in the sidebar
7. Each chat has its **own isolated PDF context**

---

## 🎯 Example Questions to Ask

After uploading a research paper:
- *"What is the main contribution of this paper?"*
- *"What methodology did the authors use?"*
- *"What were the key experimental results?"*
- *"What are the limitations of this approach?"*
- *"How does this compare to previous work?"*

---

## 🗺️ Roadmap

- [x] ChatGPT-style multi-chat interface
- [x] Per-chat isolated FAISS vector store
- [x] Persistent chat history (survives refresh)
- [x] LLM-as-a-Judge evaluation system
- [x] Deploy on Streamlit Cloud
- [x] Docker containerization & Docker Hub
- [ ] Multi-PDF support per chat
- [ ] Export chat history as PDF
- [ ] Support for images and tables in papers
- [ ] RAGAS evaluation framework integration

---

## 👨‍💻 Author

**Durgesh Kumar**
- GitHub Profile: [@Durgesh83kumar](https://github.com/Durgesh83kumar)
- Project Link: [research-paper-assistant](https://github.com/Durgesh83kumar/research-paper-assistant)
- Docker Hub: [durg1234/research-paper-assistant](https://hub.docker.com/r/durg1234/research-paper-assistant)

---

## 📄 License

MIT License — feel free to use this project for learning and portfolio purposes.

---

⭐ **If you found this useful, please star the repository!**