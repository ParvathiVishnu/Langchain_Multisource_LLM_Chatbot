import os
import torch
import tempfile
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
IS_CAUSAL_LM = True

@st.cache_resource(show_spinner=False)
def load_llm_and_embeddings():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if IS_CAUSAL_LM:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=512,
            repetition_penalty=1.15,
            do_sample=True,
            top_p=0.92,
            top_k=50
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=256,
            do_sample=False
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embed_model

llm, embed_model = load_llm_and_embeddings()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant. Using only the information in the context below, answer the user's question in a clear, concise, and non-repetitive way.
If the answer involves lists or categories, present them as bullet points.
If the context does not contain the answer, respond strictly with:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def remove_repeats(text):
    seen = set()
    result = []
    for sentence in text.split('. '):
        s = sentence.strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return '. '.join(result)

def extract_answer_only(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()

def build_vectorstore_and_qa(docs, embed_model, llm, prompt_template, k=5):
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embed_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt_template})
    return qa, chunks

def get_clean_context(chunks, n=2):
    seen = set()
    context_list = []
    for chunk in chunks[:n]:
        content = chunk.page_content.strip()
        if content not in seen:
            seen.add(content)
            context_list.append(content)
    return "\n\n".join(context_list)

def get_dynamic_page_content(url, wait_time=5):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    driver.implicitly_wait(wait_time)
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text if len(text) > 100 else None

def is_error_content(text):
    error_keywords = ["this site can’t be reached", "err_http2_protocol_error", "temporarily down", "not found", "error", "unreachable", "could not connect"]
    return any(keyword in text.lower() for keyword in error_keywords)

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Source Chatbot", layout="wide")
st.title("📙 Multi-Source Chatbot with LangChain")

source = st.tabs(["Wikipedia", "PDF", "arXiv", "Website"])

with source[0]:
    topic = st.text_input("Wikipedia Topic:", key="wiki_topic")
    question = st.text_input("Your Question:", key="wiki_question")
    if st.button("Get Answer", key="wiki_btn") and topic.strip() and question.strip():
        wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
        docs = wiki.load(query=topic)
        if docs:
            qa, chunks = build_vectorstore_and_qa(docs, embed_model, llm, prompt_template)
            context = get_clean_context(chunks)
            prompt = prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            answer = extract_answer_only(response)
            st.success(remove_repeats(answer))

with source[1]:
    uploaded_file = st.file_uploader("Upload PDF:", type=["pdf"])
    question = st.text_input("Your Question:", key="pdf_question")
    if uploaded_file and question.strip():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        qa, chunks = build_vectorstore_and_qa(docs, embed_model, llm, prompt_template)
        context = get_clean_context(chunks)
        prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(prompt)
        answer = extract_answer_only(response)
        st.success(remove_repeats(answer))

with source[2]:
    query = st.text_input("arXiv Search Query:", key="arxiv_query")
    question = st.text_input("Your Question:", key="arxiv_question")
    if st.button("Get Answer", key="arxiv_btn") and query.strip() and question.strip():
        arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=40000)
        result_str = arxiv.run(query)
        if result_str:
            from langchain.schema import Document
            docs = [Document(page_content=result_str)]
            qa, chunks = build_vectorstore_and_qa(docs, embed_model, llm, prompt_template)
            context = get_clean_context(chunks)
            prompt = prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            answer = extract_answer_only(response)
            st.success(remove_repeats(answer))

with source[3]:
    url = st.text_input("Website URL:", key="web_url")
    question = st.text_input("Your Question:", key="web_question")
    if st.button("Get Answer", key="web_btn") and url.strip().startswith("http") and question.strip():
        text = get_dynamic_page_content(url)
        if text and not is_error_content(text):
            from langchain.schema import Document
            docs = [Document(page_content=text)]
            qa, chunks = build_vectorstore_and_qa(docs, embed_model, llm, prompt_template)
            context = get_clean_context(chunks)
            prompt = prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            answer = extract_answer_only(response)
            st.success(remove_repeats(answer))
