# causal-nlp-extraction

## Introduction

In this preliminary portion of our codebase, we aim to use **OneKE** to extract knowledge from a paper of our choice. We selected a paper on **financial statement analysis using LLMs** as our test case. The paper is located in the `FinancialPapers` folder.

To get started, run the **`CheckpointTest.py`** file in the main folder. The baseline model we chose is **`meta-llama/Llama-3.1-8B-Instruct`** from the Hugging Face API, selected because it is open source and easy to access.  

> ⚠️ **Important:** To use Llama, log in to Hugging Face, create an access token with **read** permissions, and make sure it is available in your environment as `HF_TOKEN`.

We read the text from the PDF using **`PyPDF2`** and specify a task. In our example, the task is **`Triple2KG`** (short for *Triple to Knowledge Graph*), which allows us to convert the extracted triples into a knowledge graph.  

We define our **constraints** (subject, relation, and object triples for `Triple2KG`) and call the pipeline. The results are then stored in:

```text
Financial Statement Analysis Extraction.json