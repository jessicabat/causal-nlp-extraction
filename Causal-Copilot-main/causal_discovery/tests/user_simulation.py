#!/usr/bin/env python
import os
import re
import json
import pandas as pd
import numpy as np
import random  # Added to support diverse, randomized fallback generation
from llm import LLMClient
import argparse

BASE_MODEL = "gpt-4o"

def simulate_user_query(args):
    """
    Leverage LLM to generate simulated user queries for causal discovery and corresponding global_state statistics,
    then generate fake datasets based on the simulated statistics (sample_size, feature_number, and data_type) for each simulation.
    Returns:
        A list of dictionaries, where each dictionary has the following keys:
          - "initial_query": the simulated user query string,
          - "statistics": dictionary of simulated statistics,
          - "selected_algorithm": the algorithm chosen in simulation (initially None),
          - "fake_data": a pandas DataFrame containing the generated fake dataset.
    """
    
    # Load the high-performance structured meta-prompt from file
    meta_prompt_file = "user_simulation_meta_prompt.txt"
    try:
        with open(meta_prompt_file, "r") as f:
            meta_prompt = f.read()
    except Exception as e:
        meta_prompt = (
            "High-Performance Structured Meta-Prompt for User Simulation:\n"
            "Please generate a JSON object with the required keys as specified."
        )
    
    # Initialize the OpenAI client with the given credentials
    # Load pre-defined application fields
    application_fields_file = "application_fields.json"
    try:
        with open(application_fields_file, "r") as f:
            application_fields = json.load(f)
    except Exception as e:
        application_fields = {}

    meta_prompt = meta_prompt.replace("[application_fields]", json.dumps(application_fields))
    meta_prompt = meta_prompt.replace("[num_users]", str(args.num_users))

    client = LLMClient(args)
    
    # Call the LLM using the meta-prompt to simulate a user query and statistics
    response = client.chat_completion(
        prompt=meta_prompt,
        system_prompt="You are a helpful assistant for simulating user queries.",
        json_response=True
    )
    
    # Extract the response content (this is expected to be a JSON string representing an array of simulations)
    try:
        simulation_info = response
        if not isinstance(simulation_info, list):
            simulation_info = [simulation_info]
    except Exception as e:
        # Fallback simulated info if LLM parsing fails. Enhanced with random diversity.
        simulation_info = []
        for _ in range(3):  # Generate 3 fallback simulation entries
            possible_queries = [
                "selected algorithm: FGES; filter: continuous",
                "I need to explore potential causal relationships in heterogeneous data.",
                "Looking for causal discovery methods tailored for non-linear data.",
                "Please help analyze time-series data for underlying causal effects.",
                "Assess causal effects with consideration for latent variables and non-Gaussian errors."
            ]
            random_query = random.choice(possible_queries)
            statistics = {
                "linearity": random.choice([True, False]),
                "gaussian_error": random.choice([True, False]),
                "alpha": 0.05,
                "heterogeneous": random.choice([True, False]),
                "domain_index": None,
                "sample_size": random.choice([100, 500, 1000, 2000]),
                "feature_number": random.randint(5, 30),
                "data_type": random.choice(["Continuous", "Mixed"]),
                "domain_knowledge": random.choice([None, "Basic domain knowledge", "Advanced expert level"])
            }
            simulation_info.append({
                "initial_query": random_query,
                "statistics": statistics,
                "selected_algorithm": None  # Placeholder for consistency
            })
    
    # Generate fake data based on the simulated statistics for each simulation entry
    for sim in simulation_info:
        stats = sim.get("statistics", {})
        sample_size = stats.get("sample_size", 100)
        feature_number = stats.get("feature_number", 10)
        data_type = stats.get("data_type", "Continuous")
        
        if data_type == "Continuous" or data_type == "Mixed":
            fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                     columns=[f"feature_{i}" for i in range(feature_number)])
        elif data_type == "Category":
            fake_data = pd.DataFrame({
                f"feature_{i}": np.random.choice(["A", "B", "C"], size=sample_size)
                for i in range(feature_number)
            })
        elif data_type == "Time-series":
            dates = pd.date_range(start="2020-01-01", periods=sample_size, freq="D")
            fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                     index=dates,
                                     columns=[f"feature_{i}" for i in range(feature_number)])
        else:
            fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                     columns=[f"feature_{i}" for i in range(feature_number)])
        sim["fake_data"] = fake_data
    return simulation_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="User Simulation - LLM based user interaction simulation")
    parser.add_argument('--data-file', type=str, default="data/dataset/sachs/sachs.csv", help='Path to dataset file')
    parser.add_argument('--initial_query', type=str, default="selected algorithm: FGES; filter: continuous", help='Initial algorithm query')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (skip heavy preprocessing)')
    # Modes: normal, skip, simulate
    parser.add_argument('--mode', type=str, choices=["normal", "skip", "simulate"], default="simulate", help='Mode of operation')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Turn on metrics evaluation if ground truth is available')
    # Additional arguments required by the global initialization
    parser.add_argument('--organization', type=str, default="org-Xa9VGT8plP28JBRKtqBRjG5l")
    parser.add_argument('--project', type=str, default="proj_t78xpJomilJJu5qtuOK4vWfR")
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--simulation_mode', type=str, default="offline")
    parser.add_argument('--data_mode', type=str, default="real")
    parser.add_argument('--demo_mode', type=bool, default=False)
    parser.add_argument('--num_users', type=int, default=10)
    # Quick test of simulation output
    args = parser.parse_args()
    args.apikey = os.getenv("OPENAI_API_KEY")
    simulated_info = simulate_user_query(args)
    print("Simulated User Query Info:")
    # Use default=str in json.dumps to correctly display non-serializable objects like Timestamp or DataFrame content summary.
    for sim in simulated_info:
        sim['initial_query'] = json.dumps(sim['initial_query'], indent=4, default=str)
        print(sim['initial_query'])
        print(sim['statistics'])
