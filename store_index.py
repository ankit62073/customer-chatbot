from src.helper import load_csv_file, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

path = "data/Large_Website_Development_Agency_Chatbot_Dataset.csv"
extracted_data = load_csv_file(path)
embeddings = download_hugging_face_embeddings()




index_name="customer-chatbot"

#Creating Embeddings for documents
docsearch = PineconeVectorStore.from_documents(documents=extracted_data,
    embedding=embeddings,
    index_name=index_name)