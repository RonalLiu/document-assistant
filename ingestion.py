import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from constants import INDEX_NAME

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs():
    loader = ReadTheDocsLoader(
        path="docs/langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8"
    )
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"].replace("''", "/")

        new_url = old_path.replace("docs/langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pincone")
    embedding = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embedding, index_name=INDEX_NAME)
    print("***Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
