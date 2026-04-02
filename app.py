from flask import Flask, render_template, request, redirect, url_for, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from functools import wraps
from werkzeug.utils import secure_filename
import sqlite3
import uuid
import os

app = Flask(__name__)

# 🔥 Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "app_data.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            username TEXT PRIMARY KEY,
            full_name TEXT,
            mobile_no TEXT,
            email TEXT,
            gender TEXT,
            birthdate TEXT,
            photo_filename TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def get_user_profile(username):
    conn = get_db_connection()
    profile = conn.execute(
        "SELECT * FROM user_profiles WHERE username = ?",
        (username,)
    ).fetchone()

    if profile is None:
        conn.execute(
            """
            INSERT INTO user_profiles (username, full_name, mobile_no, email, gender, birthdate, photo_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, username, "", "", "", "", "")
        )
        conn.commit()
        profile = conn.execute(
            "SELECT * FROM user_profiles WHERE username = ?",
            (username,)
        ).fetchone()

    conn.close()
    return dict(profile)


def get_profile_photo_url(profile):
    photo_filename = profile.get("photo_filename", "")
    if not photo_filename:
        return ""
    return url_for("static", filename=f"uploads/{photo_filename}")


def allowed_image_file(filename):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_IMAGE_EXTENSIONS


def save_profile_photo(file_storage):
    filename = secure_filename(file_storage.filename)
    extension = filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{extension}"
    target_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file_storage.save(target_path)
    return unique_filename


init_db()

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
def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authenticated"):
        return redirect(url_for("index"))

    error_message = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == APP_USERNAME and password == APP_PASSWORD:
            session["authenticated"] = True
            session["username"] = username
            get_user_profile(username)
            return redirect(url_for("index"))

        error_message = "Invalid username or password. Please try again."

    return render_template("login.html", error_message=error_message)


@app.route("/")
@login_required
def index():
    username = session.get("username", "User")
    profile = get_user_profile(username)
    display_name = profile.get("full_name", "").strip() or username
    return render_template(
        "chat.html",
        username=display_name,
        profile_photo_url=get_profile_photo_url(profile)
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    username = session.get("username", "User")
    profile_data = get_user_profile(username)
    success_message = ""
    error_message = ""

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        mobile_no = request.form.get("mobile_no", "").strip()
        email = request.form.get("email", "").strip()
        gender = request.form.get("gender", "").strip()
        birthdate = request.form.get("birthdate", "").strip()

        photo_filename = profile_data.get("photo_filename", "")
        uploaded_photo = request.files.get("photo")

        if uploaded_photo and uploaded_photo.filename:
            if not allowed_image_file(uploaded_photo.filename):
                error_message = "Only PNG, JPG, JPEG, GIF, and WEBP images are allowed."
            else:
                photo_filename = save_profile_photo(uploaded_photo)

        if not error_message:
            conn = get_db_connection()
            conn.execute(
                """
                UPDATE user_profiles
                SET full_name = ?, mobile_no = ?, email = ?, gender = ?, birthdate = ?, photo_filename = ?, updated_at = CURRENT_TIMESTAMP
                WHERE username = ?
                """,
                (full_name, mobile_no, email, gender, birthdate, photo_filename, username)
            )
            conn.commit()
            conn.close()
            success_message = "Profile updated successfully."
            profile_data = get_user_profile(username)

    return render_template(
        "profile.html",
        username=username,
        profile=profile_data,
        profile_photo_url=get_profile_photo_url(profile_data),
        success_message=success_message,
        error_message=error_message
    )


@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")

@app.route("/get", methods=["POST"])
def chat():
    if not session.get("authenticated"):
        return "Session expired. Please sign in again.", 401

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