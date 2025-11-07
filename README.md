# causal-nlp-extraction

### Workflow Overview
PDF Paper/Text Data → OneKE Triple Extraction → JSON Output → CSV Conversion → Causal Copilot → Causal Analysis

## Introduction
In this preliminary portion of our codebase, we aim to use **OneKE** to extract knowledge from a paper of our choice. We selected the paper **Financial Statement Analysis with Large Language Models** as our test case, which can be found in the `FinancialPapers` folder.

To get started, run the **`CheckpointTest.py`** file in the main folder. The baseline model we chose is [**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) from the Hugging Face API, selected because it is open source and easy to access.  

To replicate our results, first enter the **OneKE** directory and create a local conda environment and install the required packages listed in `requirements.txt`.

```text
conda create -n oneke python=3.9
conda activate oneke
pip install -r requirements.txt
```

> ⚠️ **Important:** To use Llama, log in to Hugging Face, create an access token with **read** permissions, and make sure it is available in your environment as `HF_TOKEN`.

We read the text from the PDF using **`PyPDF2`** and specify a task. In our example, the task is **`Triple2KG`** (short for *Triple to Knowledge Graph*), which allows us to convert the extracted triples into a knowledge graph. 

We define our **constraints** (subject, relation, and object triples for `Triple2KG`) and call the pipeline. The results are then stored as:

```text
Financial Statement Analysis Extraction.json
```

This file will be generated in the project’s root directory after extraction completes.


## Postprocessing and Integration with Causal Copilot
Because **Causal Copilot** takes .csv files as input rather than .json files, we will convert the extracted data to a structured .csv format that can be passed to **Causal Copilot** for causal discovery and inference. The resulting CSV can then be uploaded to **Causal Copilot** to explore causal relationships within the extracted knowledge.