from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# HuggingFace LLM
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

app = Flask(__name__)

# Load env
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone vector store
index_name = "medical-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ✅ HuggingFace LLM 
pipe = pipeline(
    "text-generation", 
    model="google/flan-t5-base",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ QA Chain (stable)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = qa_chain.invoke(msg)
    answer = response["result"]

    print("Bot:", answer)
    return str(answer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)