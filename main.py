from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shelve
from datetime import datetime, timedelta
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load .env
load_dotenv()

# ENV variables
astra_db_config = {
    "ASTRA_DB_ID": os.getenv("ASTRA_DB_ID"),
    "ASTRA_DB_REGION": os.getenv("ASTRA_DB_REGION"),
    "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
    "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),
}
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and process documents
loader = TextLoader("FAQ.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = AstraDBVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    namespace="default_keyspace",
    collection_name="faq_qa",
    astra_db_id=astra_db_config["ASTRA_DB_ID"],
    astra_db_region=astra_db_config["ASTRA_DB_REGION"],
    astra_db_application_token=astra_db_config["ASTRA_DB_APPLICATION_TOKEN"],
    astra_db_keyspace=astra_db_config["ASTRA_DB_KEYSPACE"],
    astra_db_api_endpoint=astra_db_config["ASTRA_DB_API_ENDPOINT"]
)
retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
{context}

---

Given the context above, answer the question as best as possible.

Question: {question}

Answer:
"""
)

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4.1-nano",
    openai_api_key=openai_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
)

app = FastAPI()

class Query(BaseModel):
    message: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chat Endpoint with 10-queries/day limit ---
@app.post("/chat")
async def chat(query: Query, request: Request):
    client_ip = request.client.host
    today = datetime.now().strftime("%Y-%m-%d")

    with shelve.open("query_limit.db", writeback=True) as db:
        user_data = db.get(client_ip, {"date": today, "count": 0})

        # Reset count if it's a new day
        if user_data["date"] != today:
            user_data = {"date": today, "count": 0}

        if user_data["count"] >= 10:
            return {
                "error": "Query limit reached",
                "detail": "You’ve used all 10 queries for today. Try again tomorrow."
            }

        try:
            result = qa_chain.invoke({"query": query.message})
            user_data["count"] += 1
            db[client_ip] = user_data
            return {
                "response": result["result"],
                "queries_left": 10 - user_data["count"]
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": "Internal Server Error", "detail": str(e)}

# --- Serve index.html ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return Path("index.html").read_text()
