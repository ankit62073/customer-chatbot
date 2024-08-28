import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


#Extract data from the PDF
def load_csv_file(path):
    data = pd.read_csv(path)

    # Convert the columns to lists
    queries = data['User Query'].tolist()
    responses = data['Response'].tolist()

    # Create combined documents for each query and response
    combined_documents = [
        Document(
            page_content=f"Query: {queries[i]} Response: {responses[i]}",
            metadata={"index": i, "type": "query-response"}
        )
        for i in range(len(queries))
    ]

    return combined_documents




#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings