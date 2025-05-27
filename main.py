
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
import requests
import os
import nest_asyncio
import json
import re
from dotenv import load_dotenv

# Apply fix for nested async loop
nest_asyncio.apply()

# Load .env file
load_dotenv()

# Get API key from environment
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
parser = LlamaParse(api_key=LLAMAPARSE_API_KEY)
# parser = LlamaParse(api_key=LLAMAPARSE_API_KEY)
#API_URL = "http://localhost:3000/api/v1/prediction/b963a482-4fbb-430f-bc7a-ef359222ceee"
API_URL = "http://103.146.234.83:8080/api/v1/prediction/51d0320c-90a1-4a9f-a6e3-aa74f79f7eab"
app = FastAPI()

# CORS settings for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def extract_text_with_llamaparse(file_path):
#     documents = parser.load_data(file_path)
#     return "\n\n".join(doc.text for doc in documents)
# def extract_json_with_llamaparse(file_path):
#     documents = parser.load_data(file_path)
#     return documents[0].to_dict() if documents else {}
async def extract_json_with_llamaparse(file_path):
    documents = await parser.aload_data(file_path)
    return [doc.to_dict() for doc in documents]

@app.get("/")
def read_root():
    return {"message": "Resume analysis API is running"}


@app.post("/analyze-resume/")
async def analyze_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        resume_text = await extract_json_with_llamaparse(file_path)

        payload = {
            "question": f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
        }

        response = requests.post(API_URL, json=payload)
        os.remove(file_path)

        # Extract clean JSON string from markdown-wrapped code block
        response_text = response.json().get("text", "")

        # Use regex to extract the JSON inside the ```json ... ``` block
        match = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if match:
            structured_json_str = match.group(1)
            structured_data = json.loads(structured_json_str)
            return structured_data
        else:
            return {"error": "Failed to extract structured JSON from response"}

    except Exception as e:
        return {"error": str(e)}
