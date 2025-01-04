import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

def initialize_rag():
    # Initialize Pinecone and embeddings
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    index_name = "rag_index"
    try:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except Exception as e:
        print(f"Index creation skipped: {e}")

    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Pinecone(embedding=embeddings, index=index)

    # Load and process documents
    documents = []
    for file_name in os.listdir("data"):
        loader = TextLoader(os.path.join("data", file_name))
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    docs = text_splitter.split_documents(documents)

    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        index.upsert(vectors=[(str(uuid.uuid4()), vector, {"source": "file"})])

    retriever = vector_store.as_retriever(hybrid_weight=0.5)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

    return qa_chain

def query_rag(rag_pipeline, query):
    return rag_pipeline.run(query)
