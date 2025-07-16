# 🧠 Multi-Source Q&A Chatbot with LangChain + TinyLlama + FAISS

![LangChain](https://img.shields.io/badge/LangChain-🦜-blue?style=flat-square)
![Offline Capable](https://img.shields.io/badge/Offline-Ready-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=flat-square)
![License](https://img.shields.io/github/license/yourusername/langchain-chatbot?style=flat-square)

> Ask intelligent questions across multiple sources — all powered **locally** with Hugging Face, FAISS, and LangChain.

---

## ✨ Overview

A fully offline, multi-source chatbot powered by local LLMs and semantic search, built using:

- 🦙 **TinyLlama-1.1B-Chat** for question answering
- 🧠 **MiniLM** for fast and efficient embeddings
- 🔍 **FAISS** for chunk retrieval
- 🛠 **LangChain** to orchestrate everything
- 🎨 **Streamlit** for an elegant UI

---

## 🗂️ Project Structure

Langchain/
├── chatbot/
  ├──models/
│ ├── app.py # Main Streamlit app
│ ├── model_downloader.py # Downloads TinyLlama & MiniLM models
├── requirements.txt # Python dependencies
├── .env # (LangChain keys or tracing)
├── venv


---

## 📚 Supported Sources

| Source     | Description                          |
|------------|--------------------------------------|
| 📚 Wikipedia | Search by topic and query answers    |
| 📄 PDF       | Upload PDF files to extract insights |
| 📜 arXiv     | Search and analyze research papers   |
| 🌐 Website   | Scrape and query content dynamically |

---

## ⚙️ Setup Instructions

### 1. ✅ Create a Virtual Environment

```bash
cd Langchain
python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate     # macOS/Linux
