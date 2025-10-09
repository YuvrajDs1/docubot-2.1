# 🤖 DocuBot 2.1 — Conversational RAG with PDF Uploads

**DocuBot 2.1** is an intelligent, Streamlit-based chatbot that lets you **upload PDFs and chat with their content** — powered by **LangChain**, **Groq’s LLaMA-3**, and **HuggingFace embeddings**.

It builds a **Retrieval-Augmented Generation (RAG)** pipeline with persistent memory, incremental embedding, and rich source awareness — giving you fast, contextually accurate answers straight from your own documents.

---

## ✨ Features

- 📄 Upload and process one or more PDF files
- 🧩 Automatic text chunking and embedding
- 💾 Persistent vector store (Chroma DB)
- 🧠 Context-aware conversational memory
- ⚡ Incremental embedding — skips unchanged pages
- 🔍 Shows retrieved source snippets with filenames & pages
- 💬 Persistent chat history (`./sessions/`)
- 🎚 Adjustable model & temperature via sidebar
- 🗑 Clear DB & Add Documents controls
- 🔐 Secure Groq API key input

---

## 🧰 Tech Stack

| Component        | Technology                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------- |
| **Frontend**     | [Streamlit](https://streamlit.io)                                                         |
| **LLM Provider** | [Groq](https://console.groq.com/) — _LLaMA-3-8B-Instant_                                  |
| **Framework**    | [LangChain](https://www.langchain.com/)                                                   |
| **Embeddings**   | [HuggingFace MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-l6-v2) |
| **Vector Store** | [Chroma](https://www.trychroma.com/)                                                      |
| **PDF Loader**   | LangChain’s `PyPDFLoader`                                                                 |
| **Persistence**  | JSON chat history + Chroma DB                                                             |

---

## Deployment

The app can be accessed from: [text](https://docubot-21.streamlit.app/)
