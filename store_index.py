from src.helper import loadpdf, text_split, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

#  print(PINECONE_API_KEY)
#  print(PINECONE_API_ENV)

extracted_data = loadpdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "medical-chatbot"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Now use LangChain PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)