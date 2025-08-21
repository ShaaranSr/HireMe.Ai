#!/usr/bin/env python3
"""
HireMe.AI - Streamlit Application
A Streamlit app that uses LangGraph + LangChain (Google Gemini) to generate interview prep.
"""

import os
import io
import json
import mimetypes
from typing import List, Dict, Any, TypedDict, Optional, cast, Tuple
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv
import requests

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from prompt import (
    HIREME_AI_PROMPT,
    build_prompt,
    HireMeAIOutput,
    InterviewQA,
)

# Optional parsers for resume files
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None

# Additional optional PDF extractors
try:  # PyMuPDF – fast and robust text extraction
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

try:  # pdfminer.six – pure-Python fallback
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdfminer_extract_text = None  # type: ignore

# Optional OCR fallback for scanned PDFs
try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:  # pragma: no cover
    convert_from_bytes = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


class GraphState(TypedDict):
    inputs: Dict[str, Any]
    prompt: str
    output: Optional[HireMeAIOutput]


def extract_text_from_upload(upload) -> str:
    if upload is None:
        return ""

    # Gather basic descriptors
    mime = getattr(upload, "type", None)
    name = getattr(upload, "name", "uploaded_file")

    # Always read into bytes once to support multiple parsers
    raw_bytes: bytes
    try:
        if hasattr(upload, "getvalue"):
            raw_bytes = upload.getvalue()  # Streamlit UploadedFile
        else:
            raw = upload.read()
            raw_bytes = raw if isinstance(raw, bytes) else bytes(str(raw), "utf-8")
    except Exception:
        return ""

    # PDF handling with multiple fallbacks
    if (mime and "pdf" in str(mime).lower()) or name.lower().endswith(".pdf"):
        # 1) Try PyMuPDF first
        if fitz is not None:
            try:
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                text_parts: List[str] = []
                for page in doc:
                    text_parts.append(page.get_text("text"))
                doc.close()
                text = "\n".join(text_parts).strip()
                if text:
                    return text
            except Exception:
                pass

        # 2) Fallback to PyPDF
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(raw_bytes))
                text_chunks: List[str] = []
                for page in reader.pages:
                    try:
                        text_chunks.append(page.extract_text() or "")
                    except Exception:
                        continue
                text = "\n".join(text_chunks).strip()
                if text:
                    return text
            except Exception:
                pass

        # 3) Fallback to pdfminer.six (pure Python)
        if pdfminer_extract_text is not None:
            try:
                text = pdfminer_extract_text(io.BytesIO(raw_bytes)) or ""
                return text.strip()
            except Exception:
                pass

        # 4) OCR fallback for scanned PDFs if libraries available
        if convert_from_bytes is not None and pytesseract is not None:
            try:
                images = convert_from_bytes(raw_bytes, fmt="png")
                ocr_text_parts: List[str] = []
                for idx, img in enumerate(images):
                    # Limit OCR to first 5 pages to control latency
                    if idx >= 5:
                        break
                    try:
                        ocr_text_parts.append(pytesseract.image_to_string(img))
                    except Exception:
                        continue
                ocr_text = "\n".join(ocr_text_parts).strip()
                if ocr_text:
                    return ocr_text
            except Exception:
                pass

        # Nothing worked
        return ""

    # DOCX handling (paragraphs + tables)
    if (mime and "word" in str(mime).lower()) or name.lower().endswith(".docx"):
        if docx is None:
            return ""
        try:
            stream = io.BytesIO(raw_bytes)
            document = docx.Document(stream)
            paragraphs_text = [p.text for p in document.paragraphs]
            table_lines: List[str] = []
            for table in getattr(document, "tables", []):
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    table_lines.append("\t".join(cells))
            joined = "\n".join([*paragraphs_text, *table_lines]).strip()
            return joined
        except Exception:
            return ""

    # Plain text or unknown types: best-effort decode
    try:
        # Try utf-8 first, then latin-1 as a permissive fallback
        try:
            return raw_bytes.decode("utf-8")
        except Exception:
            return raw_bytes.decode("latin-1", errors="ignore")
    except Exception:
        return ""


# import os
# import io
# import json
# import mimetypes
# from typing import List, Dict, Any, TypedDict, Optional, cast, Tuple
# from urllib.parse import urlparse, parse_qs

# import streamlit as st
# from dotenv import load_dotenv
# import requests

# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import ValidationError

# from prompt import (
#     HIREME_AI_PROMPT,
#     build_prompt,
#     HireMeAIOutput,
#     InterviewQA,
# )

# Graph nodes

def node_compose_prompt(state: GraphState) -> GraphState:
    prompt_text = build_prompt(
        HIREME_AI_PROMPT,
        resume_text=state["inputs"]["resume_text"],
        story_text=state["inputs"]["story_text"],
        role=state["inputs"]["role"],
        company=state["inputs"]["company"],
        provided_questions=state["inputs"].get("provided_questions", []),
    )
    state["prompt"] = prompt_text
    return state


def node_call_model(state: GraphState) -> GraphState:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in your environment or sidebar.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_output_tokens=2048,
        google_api_key=api_key,
    )
    structured_llm = llm.with_structured_output(HireMeAIOutput)
    output = cast(HireMeAIOutput, structured_llm.invoke(state["prompt"]))
    state["output"] = output
    return state


def node_finish(state: GraphState) -> GraphState:
    return state


def compile_graph():
    graph = StateGraph(GraphState)
    graph.add_node("compose", node_compose_prompt)
    graph.add_node("call_model", node_call_model)
    graph.add_node("finish", node_finish)

    graph.add_edge(START, "compose")
    graph.add_edge("compose", "call_model")
    graph.add_edge("call_model", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


def render_qa_block(title: str, qas: List[InterviewQA]):
    if not qas:
        return
    st.subheader(title)
    for idx, qa in enumerate(qas, start=1):
        st.markdown(f"**Q{idx}. {qa.question.strip()}**")
        st.write(qa.answer.strip())
        st.divider()


class _UploadFromURL:
    def __init__(self, data: bytes, name: str, mime: Optional[str]):
        self._data = data
        self.name = name
        self.type = mime

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def _normalize_gdrive_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if "drive.google.com" not in parsed.netloc:
            return url
        # Pattern: /file/d/<id>/view
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
            file_id = parts[2]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        # Pattern: open?id=<id>
        qs = parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            file_id = qs["id"][0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url
    except Exception:
        return url


def fetch_upload_from_url(url: str) -> Optional[_UploadFromURL]:
    if not url:
        return None
    normalized = _normalize_gdrive_url(url.strip())
    try:
        headers = {"User-Agent": "HireMeAI/1.0"}
        resp = requests.get(normalized, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.content
        # Reject obvious HTML responses (e.g., permission pages)
        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type.startswith("text/html"):
            return None
        # Infer name and mime
        path_name = os.path.basename(urlparse(normalized).path) or "resume"
        if not content_type:
            guessed, _ = mimetypes.guess_type(path_name)
            content_type = (guessed or "").lower()
        # Fallback sniffing by signature
        if not content_type or content_type in ("application/octet-stream", "binary/octet-stream"):
            if data.startswith(b"%PDF-"):
                content_type = "application/pdf"
                if not path_name.lower().endswith(".pdf"):
                    path_name = f"{path_name}.pdf"
            elif data[:4] == b"PK\x03\x04":
                # Likely DOCX (zip); keep original name
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if not content_type:
            # Last resort, treat as text
            content_type = "text/plain"
        return _UploadFromURL(data, path_name, content_type)
    except Exception:
        return None


def main():
    load_dotenv()
    st.set_page_config(page_title="HireMe.AI", page_icon="🤖", layout="wide")

    st.title("HireMe.AI – Interview Preparation Coach")
    st.caption("LangGraph + LangChain (Gemini) powered. Upload your resume, share your story, and get tailored Q&A.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google API Key", type="password", help="Alternatively, set GOOGLE_API_KEY env var")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        st.info("Models: Gemini 1.5 Flash (default).")

    # Inputs
    col_left, col_right = st.columns([1, 1])

    with col_left:
        upload = st.file_uploader(
            "Upload resume/portfolio (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"]
        )
        if upload is not None and upload.size > 8 * 1024 * 1024:
            st.warning("Large file detected. Parsing may be slow.")
        resume_url = st.text_input(
            "Or paste resume URL (Google Drive/share link)",
            placeholder="https://...",
            help="If provided, this will be used instead of the uploaded file.",
        )
        story_text = st.text_area(
            "Personal story or inspiring journey", height=180, placeholder="Share your journey, challenges, and what drives you."
        )

    with col_right:
        role = st.text_input("Target job role", placeholder="e.g., Software Engineer")
        company = st.text_input("Target company", placeholder="e.g., TechCorp")
        raw_questions = st.text_area(
            "Possible interview questions (optional) – one per line",
            height=150,
            placeholder="Why do you want to work here?\nDescribe a challenging project you led.\n...",
        )

    provided_questions = [q.strip() for q in raw_questions.splitlines() if q.strip()]

    # Optional: always-visible debug preview of the last extracted resume text
    show_debug = st.checkbox("Show extracted resume text (latest)", value=False)
    if show_debug and "__last_resume_text" in st.session_state:
        latest = st.session_state["__last_resume_text"]
        st.info(f"Extracted {len(latest)} chars from last resume")
        with st.expander("Extracted resume text (latest)", expanded=False):
            st.text_area("Text preview", latest, height=250)

    if st.button("Generate Interview Prep", type="primary"):
        source_upload = upload
        if resume_url.strip():
            fetched = fetch_upload_from_url(resume_url)
            if fetched is None:
                st.error("Unable to fetch the resume from the provided URL. Check the link permissions or try downloading and uploading the file.")
                st.stop()
            source_upload = fetched

        resume_text = extract_text_from_upload(source_upload)
        # Persist for debug display outside the button block
        st.session_state["__last_resume_text"] = resume_text
        
        if resume_text.strip():
            st.info(f"Extracted {len(resume_text)} chars from resume")
            with st.expander("Extracted resume text (debug)", expanded=False):
                st.text_area("Text preview", resume_text, height=250)

            # If very little text was extracted from a provided file, proactively warn
            if source_upload is not None and len(resume_text) < 200:
                st.warning(
                    "The uploaded resume appears to contain very little extractable text. If it's a scanned PDF, try uploading a DOCX/TXT or an OCR'd PDF."
                )
        else:
            if source_upload is not None:
                st.warning(
                    "Could not extract any text from the uploaded resume. Proceeding with personal story only. If this is a scanned PDF, try a DOCX/TXT version or install Tesseract OCR."
                )
            else:
                st.info("No resume uploaded. Proceeding with personal story only.")

        if (not role) or (not company):
            st.error("Please enter both role and company.")
            st.stop()

        if not story_text.strip():
            st.error("Please provide a personal story to generate interview preparation.")
            st.stop()

        inputs: Dict[str, Any] = {
            "resume_text": resume_text,
            "story_text": story_text,
            "role": role,
            "company": company,
            "provided_questions": provided_questions,
        }

        with st.spinner("Thinking with HireMe.AI..."):
            try:
                app = compile_graph()
                result = app.invoke({"inputs": inputs})  # type: ignore[arg-type]
                output: HireMeAIOutput = result["output"]
            except ValidationError as ve:
                st.error(f"Validation error: {ve}")
                st.stop()
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.success("Interview prep generated.")

        st.markdown(f"### Tailored for {output.role} @ {output.company}")

        render_qa_block("Answers to Your Questions", output.provided_questions)
        render_qa_block("Additional Questions & Answers (3)", output.additional_questions)

        st.subheader("Summary Pitch")
        st.write(output.summary)

        # Downloads
        try:
            json_bytes = output.model_dump_json(indent=2).encode("utf-8")  # Pydantic v2
        except Exception:
            # Fallback: best-effort dict conversion
            try:
                payload = output.model_dump()  # Pydantic v2
            except Exception:
                try:
                    payload = output.dict()  # Pydantic v1
                except Exception:
                    payload = getattr(output, "__dict__", output)
            json_bytes = json.dumps(payload, indent=2).encode("utf-8")

        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="hireme_ai_output.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
