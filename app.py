# app.py
import os
import uuid
import hashlib
import json
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain / vectorstore imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# Load .env
load_dotenv()

DB_DIR = "./db"
DB_META_PATH = os.path.join(DB_DIR, "db_metadata.json")
SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_db_metadata():
    if os.path.exists(DB_META_PATH):
        try:
            with open(DB_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"hashes": []}
    return {"hashes": []}

def save_db_metadata(meta):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def save_session_history(session_id: str, history: List[dict]):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_session_history(session_id: str) -> List[dict]:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


st.set_page_config(page_title="DocuBot 2.1", page_icon="🤖", layout="wide")
st.title("🤖 DocuBot 2.1 — Improved Conversational RAG")


st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("GROQ API key", type="password")
st.sidebar.markdown("[Get a GROQ API key](https://console.groq.com/docs/models)")

model_choice = st.sidebar.selectbox(
    "Model",
    options=["llama-3.1-8b-instant"], 
    index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear vector DB (delete ./db)"):
    try:
        shutil.rmtree(DB_DIR, ignore_errors=True)
        # reset metadata
        save_db_metadata({"hashes": []})
        st.sidebar.success("Vector DB cleared.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear DB: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("DocuBot stores per-page chunk hashes to avoid re-embedding unchanged pages.")

if not api_key:
    st.warning("Enter your GROQ API key in the sidebar to use the assistant.")
    st.stop()


os.environ["GROQ_API_KEY"] = api_key
llm = ChatGroq(model=model_choice, api_key=api_key, temperature=float(temperature))


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id: str = st.session_state.session_id

if "store" not in st.session_state:
    st.session_state.store = {}


history_msgs = load_session_history(session_id)

def get_session_history_obj(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
        # Load persisted messages into the ChatMessageHistory if we have them
        for m in history_msgs:

            st.session_state.store[session_id].add_user_message(m["content"]) if m["role"] == "human" else st.session_state.store[session_id].add_ai_message(m["content"])
    return st.session_state.store[session_id]


@st.cache_resource(show_spinner=True)
def get_embeddings_instance():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

embeddings = get_embeddings_instance()

def create_or_update_vectorstore_from_uploads(uploaded_files) -> Chroma:
    """
    - If DB exists and no uploaded files, reuse it.
    - If uploaded files provided, load pages, compute page-hashes,
      only embed & add new chunks, persist DB and metadata.
    """
    meta = load_db_metadata()
    seen_hashes = set(meta.get("hashes", []))

    # If no uploads and DB exists -> load existing
    if (not uploaded_files) and os.path.exists(DB_DIR):
        try:
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        except Exception:
            # fallthrough to create a new empty store
            pass

    # If uploads provided, load pages from the files
    all_splits: List[Document] = []
    new_hashes = []
    for pdf in uploaded_files or []:
        tmp_path = os.path.join("./", pdf.name)
        with open(tmp_path, "wb") as f:
            f.write(pdf.getbuffer())
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()  # list of Document objects (one per page often)
        # Split pages into chunks if desired
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(raw_docs)
        # Add metadata: source filename & page if available
        for doc in splits:
            content = doc.page_content
            h = md5_text(content)
            if h not in seen_hashes:
                # annotate with metadata for source traceability
                source_file = getattr(doc, "metadata", {}).get("source", pdf.name) or pdf.name
                # If original loader provided page info in metadata, preserve it
                page_no = doc.metadata.get("page", None) or doc.metadata.get("page_number", None) or None
                doc.metadata["source_file"] = source_file
                if page_no is not None:
                    doc.metadata["page_number"] = page_no
                else:
                    # best-effort: leave absent
                    pass
                all_splits.append(doc)
                new_hashes.append(h)
            else:
                # skip embedding - already present
                continue


    if os.path.exists(DB_DIR):
        vect = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        if all_splits:
            vect.add_documents(all_splits)
            vect.persist()
    else:
        # create new
        vect = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_DIR) if all_splits else Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

        try:
            vect.persist()
        except Exception:
            pass


    meta_hashes = set(meta.get("hashes", []))
    meta_hashes.update(new_hashes)
    save_db_metadata({"hashes": list(meta_hashes)})

    return vect

# Upload UI
uploaded_files = st.file_uploader(
    "Upload one or more PDFs (or leave empty to reuse existing DB)",
    accept_multiple_files=True,
    type=["pdf"]
)

if st.button("Add uploaded PDFs to DB"):
    if not uploaded_files:
        st.warning("No files selected to add.")
    else:
        with st.spinner("Processing uploaded PDFs and adding new chunks..."):
            try:
                vectorstore = create_or_update_vectorstore_from_uploads(uploaded_files)
                st.success("Uploaded PDFs processed and new chunks added to vector DB (if any).")
            except Exception as e:
                st.error(f"Error processing uploads: {e}")

try:
    vectorstore = create_or_update_vectorstore_from_uploads(uploaded_files)
except Exception as e:
    st.error(f"Failed to initialize vectorstore: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Show some metrics
try:
    count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else "unknown"
except Exception:
    count = "unknown"
st.sidebar.metric("Indexed chunks (approx)", count)


contextualize_q_system_prompt = (
    "Given the chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do not answer the question; just rewrite it if needed."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

system_prompt = (
    "You are an assistant answering questions using the provided context. "
    "If you don't know, say you don't know. Keep answers concise (<= 3 sentences).\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history_obj,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


st.divider()
st.subheader("💬 Chat (conversational)")

# Load persisted simple JSON history to show UI
json_history = load_session_history(session_id)
# Show previous messages
chat_container = st.container()
with chat_container:
    for m in json_history:
        role = m.get("role", "human")
        content = m.get("content", "")
        if role == "human":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)

# Input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    json_history.append({"role": "human", "content": user_input})
    save_session_history(session_id, json_history)

 
    try:
        with st.spinner("Retrieving relevant context..."):
            retrieved_docs = retriever.get_relevant_documents(user_input)
    except Exception as e:
        retrieved_docs = []
        st.error(f"Retrieval failed: {e}")

    # Show retrieved context snippets and their source metadata
    if retrieved_docs:
        with st.expander("🔎 Retrieved context (click to view)"):
            for i, d in enumerate(retrieved_docs, start=1):
                src = d.metadata.get("source_file", "unknown")
                page = d.metadata.get("page_number", d.metadata.get("page", "unknown"))
                snippet = d.page_content[:600].replace("\n", " ")
                st.markdown(f"**{i}. Source:** `{src}` — page: `{page}`")
                st.caption(snippet + ("..." if len(d.page_content) > 600 else ""))
    else:
        st.info("No context retrieved (vector DB may be empty).")

    try:
        with st.spinner("Generating answer..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
        answer = response.get("answer", "No answer returned.")
        with st.chat_message("assistant"):
            st.write(answer)

        json_history.append({"role": "assistant", "content": answer})
        save_session_history(session_id, json_history)
    except Exception as e:
        st.error(f"Error generating answer: {e}")


st.write("---")
st.caption("DocuBot 2.1 — features: metadata-backed retrieval, incremental embedding, persisted chat history.")
