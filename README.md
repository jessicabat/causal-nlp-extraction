# causal-nlp-extraction

## Introduction
In this preliminary portion of our Codebase, we aim to just use OneKE to extract knowledge from a paper of our choice. We have chosen a paper of finantial statement analyses using LLM's as our test. The paper can be found in the 'FinancialPapers' folder.

To get started, use the 'CheckpointTest.py' file that can be found in the main folder. The baseline model that was chosen was meta-llama/Llama-3.1-8B-Instruct from the HuggingFace API due to its ease of access from being an open source model. In order to use Llama, log in to HuggingFace and create an access token, allowing READ access and pass that token into the client. 

We read in the text from the paper using PyPDF2 and specify a task. In our case, we chose Triple2KG, short for "Triple to Knowledge Graph". The use of Triple2KG will later allow us to easily create a knowledge graph from the extracted text. We specify our constraints (in the Triple2KG example it is subject, relation, object triples) and call the pipeline. The results from the extraction are then stored in the "Financial Statement Analysis Extraction.json". 