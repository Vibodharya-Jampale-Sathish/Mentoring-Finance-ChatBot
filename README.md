# 🧠 Mentoring Finance Chatbot 

This project is a conversational AI chatbot for finance-related mentoring queries. It uses **FastAPI**, **LangChain**, and **AstraDB**, with OpenAI’s GPT embeddings for intelligent context-aware responses. Designed to be deployable and embeddable, it supports both frontend integration and terminal interaction.

---

## 🚀 Features

- ⚡ FastAPI REST backend
- 🧠 LangChain-based conversation engine
- 📄 Custom finance FAQs loaded and embedded
- 🗃️ AstraDB vector store with OpenAI embeddings
- 🧾 Optional React + Tailwind frontend or Langflow chat widget
- 🔐 .env configuration for secure API keys
- 📦 Deployment-ready structure

---

## 🧩 Architecture Overview

```mermaid
graph TD
    A[User] --> B[Chat UI]
    B --> C[FastAPI Server]
    C --> D[LangChain QA]
    D --> E[OpenAI LLM]
    D --> F[AstraDB Vector Store]
    F --> D
    E --> D
    D --> G[Response]
    G --> B
```
---
## 🛠️ Project Structure
``` bash
Mentoring_Finance_Chatbot/
└── /langchain_flow              # LangChain flows or utilities (optional)/
├── app.py                 # FastAPI application
├── database.sql           # SQL schema for user and blog tables
├── readme.md              # Complete documentation
├── .env                   # Sensitive config (excluded from git)
├── /models                # Optional: Pydantic models
├── /langchain_flow        # LangChain integration
└── /static                
```
---

## 🛠️ The ENV Structure

```env
# OpenAI API
OPENAI_API_KEY=sk-...

# Astra DB Settings
ASTRA_DB_ID=your-astra-db-id
ASTRA_DB_APPLICATION_TOKEN=your-astra-token
ASTRA_DB_REGION=us-east-2
ASTRA_DB_KEYSPACE=default_keyspace
ASTRA_DB_API_ENDPOINT=https-link
```
---

## 🛠️ Technologies Used

### ⚙️ Backend Frameworks & APIs

| Technology    | Purpose                                                  |
|---------------|-----------------------------------------------------------|
| **FastAPI**   | High-performance API framework for building web services |
| **Uvicorn**   | ASGI server to run the FastAPI app                        |
| **LangChain** | Framework for LLM chaining, embedding, and memory        |
| **OpenAI API**| For generating and embedding natural language responses  |

---

### 🧠 AI & Embedding Tools

| Technology              | Purpose                                                |
|-------------------------|--------------------------------------------------------|
| **OpenAIEmbeddings**    | Converts user input into vector embeddings             |
| **SplitText**           | Breaks text into chunks before embedding               |
| **AstraDB (Vector Store)** | Stores embeddings and performs similarity search   |

---

### 🗃️ Database & Data Access

| Technology            | Purpose                                         |
|------------------------|-------------------------------------------------|
| **AstraDB (Cassandra)**| Scalable vector database for embeddings        |
| **cassandra-driver**   | Python SDK to connect to AstraDB               |

---

### 🔐 Configuration & Environment

| Technology        | Purpose                                             |
|-------------------|-----------------------------------------------------|
| **python-dotenv** | Loads environment variables from a `.env` file      |
| **.env file**     | Stores API keys and secrets securely                |

---

### 🧪 Development Tools

| Technology   | Purpose                                             |
|--------------|-----------------------------------------------------|
| **pydantic** | Data validation and typing for FastAPI models       |
| **typing**   | Python static type hints                            |
| **logging**  | Console/debug output tracking                       |

---

### 💬 Optional Frontend

| Technology               | Purpose                                                  |
|--------------------------|----------------------------------------------------------|
| **Langflow Chat Widget** | Embeds the bot using a script on HTML pages              |
| **HTML / JavaScript**    | Used to mount the chat interface (optional)              |

