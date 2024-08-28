from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


embeddings = download_hugging_face_embeddings()


index_name="customer-chatbot"

#Loading the index
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    chain_type="stuff",
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    response_text = result['source_documents'][0].page_content.split("Response:", 1)[1].strip()
    print("Response:", response_text)
    return str(response_text)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
