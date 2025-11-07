import sys
import os
from pathlib import Path

# fix path first — before importing anything else from OneKE
ROOT = Path(__file__).resolve().parent.parent  # goes from examples → OneKE
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# now these imports will work
from models import *
from pipeline import *
from huggingface_hub import InferenceClient
import json
import PyPDF2

# create HF inference client
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",  # use the instruct version
    token=os.environ["HF_TOKEN"]
)

class LlamaModel:
    def __init__(self, client):
        self.client = client
        self.name = "Llama"  # required by OneKE

    def get_chat_response(self, prompt):
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        # Extract the content
        return response.choices[0].message["content"]

# replace ChatGPT with Llama
model = LlamaModel(client)
pipeline = Pipeline(model)

# extraction configuration
Text = ""
with open("../../Financial Statement Analysis with Large Language Models.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        Text += page.extract_text() + "\n"
        
Task = "Triple2KG"
Constraint = [
    ["Company", "Executive", "Merger", "Transaction"],
    ["reports", "discloses", "increases", "decreases", "forecasts", "impacts"],
    ["FinancialMetric", "FiscalPeriod", "MarketSegment", "Regulation", "CashFlow", "Revenue", "NetIncome"]
]

result, trajectory, frontend_schema, frontend_res = pipeline.get_extract_result(
    task=Task,
    text=Text,
    constraint=Constraint,
    show_trajectory=True
)

all_data = {
    "result": result,
    "trajectory": trajectory,
    "frontend_schema": frontend_schema,
    "frontend_res": frontend_res
}

output_file = "onke_extraction_output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"Extraction results saved to {output_file}")