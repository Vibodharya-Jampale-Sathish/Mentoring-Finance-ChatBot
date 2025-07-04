from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
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

# Load environment variables
load_dotenv()

# --- ENV variables ---
astra_db_config = {
    "ASTRA_DB_ID": os.getenv("ASTRA_DB_ID"),
    "ASTRA_DB_REGION": os.getenv("ASTRA_DB_REGION"),
    "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
    "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),

}
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Document & Vector Store Setup ---
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
    astra_db_api_endpoint=astra_db_config["ASTRA_DB_API_ENDPOINT"],  # ✅ Use endpoint here
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

# --- FastAPI App ---
app = FastAPI()

# In-memory store: {ip_address: count}
query_counters: Dict[str, int] = {}

class Query(BaseModel):
    message: str

# --- Middleware for CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Limit 20 Queries per IP ---
@app.post("/chat")
async def chat(query: Query, request: Request):
    client_ip = request.client.host
    count = query_counters.get(client_ip, 0)

    if count >= 20:
        return {"error": "Query limit reached", "detail": "You have used all 20 available queries."}

    try:
        result = qa_chain.invoke({"query": query.message})
        query_counters[client_ip] = count + 1
        return {"response": result["result"], "queries_left": 20 - query_counters[client_ip]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Internal Server Error", "detail": str(e)}

# --- Serve Frontend (index.html) ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return Path("index.html").read_text()
