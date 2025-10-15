from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from PyPDF2 import PdfReader
from openai import OpenAI
from rapidfuzz import fuzz
from googleapiclient.errors import HttpError
from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import math
import io
import os
import pickle
import tempfile
import numpy as np

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# OpenAI Setup
# -----------------------------
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY_S")
)

# -----------------------------
# Google Drive Setup
# -----------------------------
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=creds)

DEMO_FOLDER_ID = "1lyKKM94QxpLf0Re76_1rGuk5gCRWcuP0"

# -----------------------------
# Google Cloud Storage Setup
# -----------------------------
# Path to your service account JSON for GCS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account_gcs.json"

# Initialize GCS client and get the bucket
gcs_client = storage.Client()
gcs_bucket_name = "krishdemochatbot"
gcs_bucket = gcs_client.bucket(gcs_bucket_name)

def upload_to_gcs(file_bytes, blob_name):
    blob = gcs_bucket.blob(blob_name)
    blob.upload_from_string(file_bytes)
    print(f"Uploaded {blob_name} to bucket {gcs_bucket.name}")

def download_from_gcs(blob_name):
    blob = gcs_bucket.blob(blob_name)
    return blob.download_as_bytes()

# -----------------------------
# PDF Utilities
# -----------------------------
def download_pdf(file_id):
    fh = io.BytesIO()
    request = drive_service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def list_pdfs(folder_id, level=0):
    results = []
    indent = "  " * level
    page_token = None

    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, webViewLink)',
            pageToken=page_token
        ).execute()

        for file in response.get('files', []):
            if file['mimeType'] == 'application/pdf':
                results.append({
                    "id": file['id'],
                    "name": file['name'],
                    "webViewLink": file.get('webViewLink', ''),
                    "parent_id": folder_id
                })
            elif file['mimeType'] == 'application/vnd.google-apps.folder':
                results.extend(list_pdfs(file['id'], level + 1))

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

    return results

def summarize_with_openai(snippet: str, query: str) -> str:
    prompt = f"""
    You are an educational assistant.

    Based on the following PDF content:
    {snippet}

    Answer concisely what this PDF is about in the context of the question: '{query}'.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# -----------------------------
# Vector Store Utilities
# -----------------------------
MAX_CHARS_PER_SNIPPET = 4000
THRESHOLD = 75

def vectorstore_exists_in_drive(pdf_name, folder_id):
    vs_name = f"{pdf_name}.pkl"
    response = drive_service.files().list(
        q=f"'{folder_id}' in parents and name='{vs_name}' and trashed=false",
        fields="files(id, name)"
    ).execute()
    return bool(response.get("files", []))

def upload_vectorstore_to_drive(vectorstore_bytes, vs_name, folder_id):
    file_metadata = {"name": vs_name, "parents": [folder_id]}
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(io.BytesIO(vectorstore_bytes), mimetype="application/octet-stream")
    drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()




# Keep track of PDFs processed in this run
processed_pdfs = set()
def create_vectorstore_for_pdf(pdf_file):
    """
    Create a vector store for a single PDF and upload it to the correct GCS bucket.
    Each chunk includes metadata: page number, chunk index, and Google Drive link.
    Skips processing if the PDF already exists in the bucket.
    Embeddings are normalized so FAISS similarity scores are in [0, 1] (cosine similarity).
    """

    pdf_name = pdf_file.get("name")
    pdf_base_name = pdf_name.replace(".pdf", "")
    pdf_link = pdf_file.get("webViewLink", "")

    # Check if PDF already exists in GCS
    pdf_blob_name = f"{pdf_base_name}/{pdf_name}"
    if gcs_bucket.blob(pdf_blob_name).exists():
        print(f"[DEBUG] PDF {pdf_name} already exists in GCS. Skipping upload.")
        return

    # Download PDF bytes from Drive
    pdf_id = pdf_file.get("id")
    if pdf_id is None:
        print(f"[DEBUG] PDF {pdf_name} has no Drive ID. Skipping.")
        return
    file_bytes = download_pdf(pdf_id)
    reader = PdfReader(io.BytesIO(file_bytes))

    # Prepare chunks with metadata
    chunks = []
    print(f"[DEBUG] Processing PDF: {pdf_name}, Total pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            cleaned_text = " ".join(
                line.strip() for line in page_text.splitlines() if line.strip()
            )

            # Semantic chunking: larger chunk size with overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=250,
                separators=["\n\n", "\n", " "]
            )
            page_chunks = text_splitter.create_documents([cleaned_text])

            for j, chunk in enumerate(page_chunks, start=1):
                chunk.metadata.update({
                    "pdf_name": pdf_name,
                    "pdf_base_name": pdf_base_name,
                    "page_number": i,
                    "chunk_index": j,
                    "pdf_link": pdf_link,
                    "chunk_length": len(chunk.page_content)
                })
                chunks.append(chunk)
                print(f"[DEBUG] Chunk added: PDF={pdf_name}, Page={i}, Chunk={j}, Text sample='{chunk.page_content[:100]}...'")

    if not chunks:
        print(f"[DEBUG] No text found in PDF {pdf_name}, skipping vector store creation.")
        return

    # Create embeddings using a strong model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY_S")
    )

    # Create FAISS vector store
    vs = FAISS.from_documents(chunks, embeddings)
    print(f"[DEBUG] FAISS vector store created for PDF {pdf_name}, Total chunks: {len(chunks)}")

    # 🔹 Normalize embeddings so FAISS returns cosine similarity in [0,1]
    if hasattr(vs, "index") and hasattr(vs.index, "normalize_L2"):
        vs.index.normalize_L2()
        print("[DEBUG] FAISS embeddings normalized (unit vectors) for cosine similarity.")

    # Inspect embedding vectors for a few chunks
    for idx, doc in enumerate(chunks[:5]):
        vec = embeddings.embed_query(doc.page_content)
        norm = sum([v**2 for v in vec])**0.5
        print(f"[DEBUG] Embedding vector sample for chunk {idx+1}: Norm={norm:.4f}, Text sample='{doc.page_content[:80]}...'")

    # Save locally and upload vector store files to GCS
    gcs_prefix_vs = f"{pdf_base_name}/vectorstore/"
    with tempfile.TemporaryDirectory() as tmp_dir:
        vs.save_local(tmp_dir)
        for filename in os.listdir(tmp_dir):
            path = os.path.join(tmp_dir, filename)
            with open(path, "rb") as f:
                blob_name = f"{gcs_prefix_vs}{filename}"
                upload_to_gcs(f.read(), blob_name)
                print(f"[DEBUG] Uploaded vector store file to GCS: {blob_name}")

    # Upload original PDF to GCS
    upload_to_gcs(file_bytes, pdf_blob_name)
    print(f"[DEBUG] Uploaded PDF to GCS: {pdf_blob_name}")
    print(f"[INFO] Vector store and PDF for {pdf_name} uploaded successfully to GCS.")






def ensure_vectorstores_for_all_pdfs(pdf_files):
    for pdf in pdf_files:
        create_vectorstore_for_pdf(pdf)

def load_vectorstore_from_gcs(gcs_prefix: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Downloads the FAISS vector store files from a GCS prefix and loads the vector store.

    Args:
        gcs_prefix (str): The folder/prefix in GCS where the vector store files are located.
        embeddings (OpenAIEmbeddings): The embeddings object to use for loading the vector store.

    Returns:
        FAISS: The loaded FAISS vector store instance.
    """
    # Initialize Google Cloud Storage client
    gcs_client = storage.Client()

    # List all blobs under the given prefix
    blobs = list(gcs_client.list_blobs(gcs_bucket_name, prefix=gcs_prefix))
    if not blobs:
        raise ValueError(f"No vector store files found in GCS for prefix '{gcs_prefix}'.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download each blob to the temp directory
        for blob in blobs:
            file_path = os.path.join(tmp_dir, os.path.basename(blob.name))
            blob.download_to_filename(file_path)

        # Load the vector store from the temp directory
        vs = FAISS.load_local(
            tmp_dir,
            embeddings,
            allow_dangerous_deserialization=True  # Safe for your own files
        )

    return vs
# -----------------------------
# API Endpoints
# -----------------------------
SIMILARITY_THRESHOLD = 0.40  # cosine similarity threshold (adjust as needed)
TOP_K = 5  # max chunks per PDF


REWRITER_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"



@app.get("/search")
async def search_pdfs(query: str = ""):
    results = []
    top_chunks = []

    # 1️⃣ List PDFs and ensure vector stores exist
    pdf_files = list_pdfs(DEMO_FOLDER_ID)
    ensure_vectorstores_for_all_pdfs(pdf_files)

    # 2️⃣ Expand the query for better retrieval
    rewritten_query_prompt = (
        f"Rephrase the following question to make it more specific for finding relevant sections in educational PDFs, "
        f"but keep all the original key words and phrases intact: {query}"
    )
    response = openai_client.chat.completions.create(
        model=REWRITER_MODEL,
        messages=[{"role": "user", "content": rewritten_query_prompt}],
        temperature=0.2
    )
    rewritten_query = response.choices[0].message.content
    print(f"Original query: {query}")
    print(f"Rewritten query for retrieval: {rewritten_query}")

    # 3️⃣ Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY_S")
    )

    # 4️⃣ Retrieve relevant chunks from each PDF
    for pdf in pdf_files:
        pdf_base_name = pdf["name"].rsplit(".", 1)[0]
        gcs_prefix = f"{pdf_base_name}/vectorstore/"

        # Load vector store
        vectorstore: FAISS = load_vectorstore_from_gcs(gcs_prefix, embeddings)

        # ⚡ Normalize embeddings in FAISS index (if needed)
        if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "normalize_L2"):
            vectorstore.index.normalize_L2()

        # Similarity search with scores (distance-based)
        docs_with_scores = vectorstore.similarity_search_with_score(rewritten_query, k=TOP_K)

        # Collect all chunks with their distance
        for doc, distance_score in docs_with_scores:
            doc.metadata.update({
                "pdf_name": pdf["name"],
                "pdf_base_name": pdf_base_name
            })
            top_chunks.append((doc, distance_score))

    # 5️⃣ Early return if no chunks found
    if not top_chunks:
        results.append({"name": "No results found", "snippet": "", "link": ""})
        return JSONResponse(results)

    # 6️⃣ Sort by ascending distance (least distance = most relevant)
    top_chunks = sorted(top_chunks, key=lambda x: x[1])[:5]  # Take the most relevant

    # 7️⃣ Prepare context for GPT answer
    context_texts = [
        f"PDF: {doc.metadata['pdf_name']}, Page: {doc.metadata['page_number']}\n{doc.page_content}"
        for doc, _ in top_chunks
    ]
    answer_prompt = f"""
You are an assistant. Answer the user question using only the following PDF chunks.
For each fact, indicate the PDF name and page number it came from.

Question: {query}
Chunks:
{chr(10).join(context_texts)}
"""

    answer_response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.2
    )
    answer_text = answer_response.choices[0].message.content

    # 8️⃣ Collect PDF links used
    used_pdfs = list({doc.metadata.get("pdf_link") for doc, _ in top_chunks if doc.metadata.get("pdf_link")})

    results.append({
        "name": "Answer",
        "snippet": answer_text,
        "link": ", ".join(used_pdfs)
    })

    return JSONResponse(results)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_chatbot_backend_2:app", host="0.0.0.0", port=8000, reload=True)
