# HireMe.AI 🤖

Streamlit app powered by LangGraph + LangChain (Google Gemini) to generate tailored interview preparation from your resume and personal story.

## ✨ What it does
- **Upload Resume/Portfolio (PDF, DOCX, TXT)**
- **Provide Personal Story** (your journey, challenges, strengths)
- **Target Role & Company**
- **Optional Interview Questions** (one per line)

HireMe.AI will:
- Craft persuasive answers to your provided questions (4–6 sentences each)
- Generate 5 additional realistic questions with answers
- Summarize your candidate pitch (2–3 sentences)

All content is tailored to the role and company, with confident, human tone.

## 🧱 Tech
- **App**: Streamlit
- **LLM**: Google Gemini via `langchain-google-genai`
- **Orchestration**: LangGraph + LangChain
- **(Optional) Tools**: `langchain-mcp-adapters` (configure MCP separately)

## 🚀 Quickstart

### 1) Clone / open the project directory
```bash
cd "F:\NxtWave\MCP AI WORKSHOP\HireMe.AI"
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Set your Google API key
- Get an API key for Google Generative AI (Gemini)
- Set it as an environment variable:
```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY="YOUR_KEY_HERE"
```
Or create a `.env` file with:
```
GOOGLE_API_KEY=YOUR_KEY_HERE
```
You can also paste the key in the app sidebar.

### 5) Run the app
```bash
streamlit run app.py
```

Open the displayed local URL in your browser.

## 🧑‍💻 Usage
1. Upload your resume (PDF/DOCX/TXT)
2. Paste your personal story
3. Enter role and company
4. Optionally add possible interview questions (one per line)
5. Click “Generate Interview Prep”
6. Review tailored answers, additional Q&A, and summary
7. Download JSON if needed

## 🔐 Environment
- `GOOGLE_API_KEY` (required): Gemini access
- `.env` supported via `python-dotenv`

## 🧩 Optional: MCP tool integration
This project includes `langchain-mcp-adapters` in requirements. If you have MCP tool servers set up, you can integrate tools into your chains/graphs for retrieval or utilities. This sample app runs without MCP; extend the graph to register tools as needed.

## 📝 Notes
- PDF/DOCX parsing is best-effort using `pypdf` and `python-docx`
- The app truncates very long text to keep prompts within model limits
- Default model: `gemini-1.5-flash`

## 🧪 Development
```bash
# Lint/type check (optional)
# pip install black flake8 mypy
# black . && flake8
```

---
Built with ❤️ using Streamlit, LangGraph, and Gemini.
