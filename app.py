from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# 🔥 Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Debug check (optional)
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env")

# 🔥 Embeddings
embeddings = download_hugging_face_embeddings()

# 🔥 Pinecone setup
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 🔥 FIXED MODEL (OpenRouter)
chatModel = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",  # or mistral / gpt-4o-mini
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)

# 🔥 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 🔥 RAG Chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 🔥 Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form["msg"]
        print("User:", user_input)

        response = rag_chain.invoke({"input": user_input})

        print("Response:", response["answer"])
        return str(response["answer"])

    except Exception as e:
        print("Error:", str(e))
        return "❌ Error occurred: " + str(e)

# 🔥 Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)