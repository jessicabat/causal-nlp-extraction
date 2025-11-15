# causal-nlp-extraction

### Workflow Overview
PDF Paper/Text Data → OneKE Triple Extraction → Knowledge Graph → CSV Conversion → Causal Copilot → Causal Analysis

## Introduction
In this preliminary portion of our codebase, we aim to use `OneKE` to extract knowledge from a paper of our choice. We selected the paper `Financial Statement Analysis with Large Language Models` as our test case, which can be found in the `FinancialPapers` folder.

To replicate our results, first clone our repository and ensure you have docker desktop installed. Go through each `.yaml` file and change the model to your desired model, extraction mode and constraints to your desired extraction mode which can be found in the `OneKE/src/config.yaml` file and constraints which you can specify. In the `construct` section, ensure you have your own instance of `Neo4j` running either locally or remotely through `Neo4j AuraDB`. This can be done through docker (locally) or online (remotely). Enter in the corresponding `url` and `password` for your own instance.

#### Neo4j Aura Example Construct Section:
```yaml
construct: # Need this for constructing Knowledge Graph
  database: Neo4j 
  url: neo4j+s://<database-id>.databases.neo4j.io
  username: neo4j # your database username.
  password: "<database_password>" # your database password.
  graph_name: MyKG # optional, a name for your KG
```

#### Neo4j Local Example Construct Section:
```yaml
construct: # Need this for constructing Knowledge Graph
  database: Neo4j 
  url: bolt://localhost:7687 # your database URL，Neo4j's default port is 7687.
  username: neo4j # your database username.
  password: "<database_password>" # your database password.
  graph_name: MyKG # optional, a name for your KG
```

Enter the `main` directory of our repo and run the commands below to pull our docker image and create a container for processing. 

> ⚠️ **Important:** When pulling the image from Github, there are two versions: One that is compatible for **Windows** (`amd64`) and one that is compatible for **Mac** (`arm64`). To specify which version you want to pull, ensure the **tag** is either: `:latest` for `arm64`, or `:amd64` for `amd64`.

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

#### For users with an NVIDIA GPU, you can use the NVIDIA runtime to leverage GPU acceleration:

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it --gpus all \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

After running the container and entering the `causal-nlp-extraction` directory, run the **`run.py`** file in the `OneKE/src/` folder. The baseline model we chose is [**Qwen/Qwen2.5-0.5B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) from the Hugging Face API, selected because it is open source, easy to access, and quick for extraction. 

> ⚠️ **Important:** To use QWEN, log in to Hugging Face, create an access token with **read** permissions. Use the command below and enter in your key to gain access to Hugging Face:

```bash
huggingface-cli login
```

After Hugging Face recognizes your key, run this command below to start knowledge extraction:

```bash
python OneKE/src/run.py --config <yaml_file>
```

When the process finishes, you should see the knowledge extracted in your terminal, and further pushed to `Neo4j` to create a knowledge graph in the explore tab of your instance.

## Postprocessing and Integration with Causal Copilot
Because **Causal Copilot** takes .csv files as input rather than .json files, we will convert the extracted data to a structured .csv format that can be passed to **Causal Copilot** for causal discovery and inference. The resulting CSV can then be uploaded to **Causal Copilot** to explore causal relationships within the extracted knowledge.