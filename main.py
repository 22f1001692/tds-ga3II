from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
import google.generativeai as genai
import os


app = FastAPI()

# --- 1. Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows the Cloudflare grader to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Define Input and Output Schemas ---
class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

# Initialize your LLM client 
# Ensure your API key is set in your terminal: export OPENAI_API_KEY="your-key-here"

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-omni-flash-preview')

# Inside your /extract endpoint:


# --- 3. The Extraction Endpoint ---
@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: InvoiceRequest):
    # The Prompt does the heavy lifting for Date formatting and Subtotal/Tax logic
    system_prompt = """
    You are an expert accounting assistant. Extract the following fields from the raw invoice text provided by the user.
    
    Rules for extraction:
    - invoice_no: The invoice reference or number.
    - date: Convert the date strictly to ISO format YYYY-MM-DD. 
    - vendor: The name of the vendor or issuing company.
    - amount: The subtotal amount BEFORE tax. Extract as a float.
    - tax: The tax amount (e.g., GST/IGST) ONLY. Extract as a float. Do not include the total.
    - currency: The 3-letter currency code (e.g., INR, USD).
    
    Return ONLY a valid JSON object with the exact keys: "invoice_no", "date", "vendor", "amount", "tax", "currency".
    If any field cannot be found, set its value to null.
    """

    try:
        # Call the LLM with JSON mode enabled
        response = model.generate_content(
            f"{system_prompt}\n\nRaw Invoice Text:\n{request.invoice_text}",
            generation_config={"response_mime_type": "application/json"}
        )
        extracted_data = json.loads(response.text)
        
        # Pydantic (InvoiceResponse) will automatically validate and format this return
        return extracted_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
