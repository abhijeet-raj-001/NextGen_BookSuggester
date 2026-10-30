import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def get_rag_chain(persist_dir="./chroma_books_db"):
    # 1. Embeddings: Runs locally (still using Hugging Face for free local vector embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Retriever
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. LLM Setup (OpenRouter serving Qwen)
    or_token = os.getenv("OPENROUTER_API_KEY")
    if not or_token:
        raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")
    
    # We use ChatOpenAI but point the base URL to OpenRouter
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=or_token,
        model_name="openrouter/free", # Automatically routes to an active free model
        temperature=0.3,
        max_tokens=512,
    )

    # 4. Prompt: ChatPromptTemplate handles the specific model formatting natively
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Lumina, an expert literary concierge. Recommend and discuss books using ONLY the supplied context.\nIf the answer cannot be determined from the context, gently mention that the title is not in your current catalog.\n\nRetrieved Catalog Context:\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 5. RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever