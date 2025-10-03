"""
Streamlit RAG App
=================

This app provides a Streamlit interface for:
1. Uploading PDFs and triggering ingestion into a RAG pipeline.
2. Asking natural language questions against the ingested PDFs.
3. Querying Inngest for responses, sources, and context chunks.

Features:
- Custom theme styling with teal, honey, lilac, and cream.
- PDF upload handling and event sending to Inngest.
- Query submission form with result rendering.
"""

# -------------------- Imports --------------------
import asyncio
import os
import time
from pathlib import Path

import requests
import streamlit as st
import inngest
from dotenv import load_dotenv


# -------------------- Env & Config --------------------
load_dotenv()

st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered",
)


# -------------------- Theme & Styling --------------------
PRIMARY = "#008080"      # Teal
SECONDARY = "#F4D06F"    # Honey
ACCENT = "#C39BD3"       # Lilac
BG = "#FFF7E6"           # Cream
TEXT = "#333333"

st.markdown(f"""
<style>
    .main {{
        background-color: {BG};
        color: {TEXT};
    }}
    .stButton>button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 8px;
    }}
    .stButton>button:hover {{
        background-color: {ACCENT};
        color: white;
    }}
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {{
        border-radius: 6px;
        border: 1px solid {PRIMARY};
        padding: 6px;
    }}
    .stSpinner>div {{
        color: {PRIMARY};
    }}
    .stFileUploader>div>div {{
        background-color: {SECONDARY};
        border-radius: 8px;
        padding: 10px;
    }}
</style>
""", unsafe_allow_html=True)


# -------------------- Inngest Client --------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """
    Get a cached instance of the Inngest client.
    """
    return inngest.Inngest(app_id="rag_app", is_production=False)


# -------------------- PDF Handling --------------------
def save_uploaded_pdf(file) -> Path:
    """
    Save an uploaded Streamlit file uploader object to disk.

    Args:
        file: Streamlit UploadedFile instance.

    Returns:
        Path to the saved PDF file.
    """
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> str:
    """
    Send an ingestion event to Inngest for a PDF.

    Args:
        pdf_path: Path to the saved PDF.

    Returns:
        Event ID string.
    """
    client = get_inngest_client()
    res = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(pdf_path.resolve()), "source_id": pdf_path.name},
        )
    )
    print("DEBUG Inngest ingest send() result:", res)
    return res[0]  # Inngest typically returns a list of event IDs


# -------------------- Upload Section --------------------
st.markdown(f"<h1 style='color:{PRIMARY}'>📄 Upload a PDF to Ingest</h1>", unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        event_id = asyncio.run(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"✅ Triggered ingestion for: {path.name} (event: {event_id})")
    st.caption("You can upload another PDF if you like.")

st.divider()


# -------------------- Query Section --------------------
st.markdown(f"<h1 style='color:{ACCENT}'>💬 Ask a question about your PDFs</h1>", unsafe_allow_html=True)


async def send_rag_query_event(question: str, top_k: int) -> str:
    """
    Send a query event to Inngest.

    Args:
        question: User's query string.
        top_k: Number of chunks to retrieve.

    Returns:
        Event ID string.
    """
    client = get_inngest_client()
    res = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    print("DEBUG Inngest query send() result:", res)
    return res[0]


def _inngest_api_base() -> str:
    """
    Return the base URL for the Inngest API.
    """
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    """
    Fetch run data for a given event ID from Inngest API.
    """
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    """
    Wait for a run to complete and return its output.

    Args:
        event_id: Inngest event ID.
        timeout_s: Max time to wait before giving up.
        poll_interval_s: Interval between polling attempts.

    Returns:
        Output dictionary from the completed run.
    """
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# -------------------- Query Form --------------------
with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.markdown(f"<h3 style='color:{PRIMARY}'>Answer</h3>", unsafe_allow_html=True)
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
