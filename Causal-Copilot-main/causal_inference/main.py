import os
import numpy as np
import pandas as pd 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pydantic import BaseModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from causal_inference.inference import Analysis
from causal_inference.help_functions import LLM_parse_query
from utils.logger import logger

def main(global_state, args):
    """
    Modify the main function to call attribute_anomalies and save the results in ./auto_mpg_output.
    """
    logger.header("Causal Analysis Demo - Sachs Dataset")
    analysis = Analysis(global_state, args)
    message = input("Is there any causal analysis problem I can help you? \n")

    class InfList(BaseModel):
                tasks: list[str]
                descriptions: list[str]
                key_node: list[str]
    columns = global_state.user_data.processed_data.columns
    with open('causal_inference/context/query_prompt.txt', 'r') as file:
        query_prompt = file.read()
        query_prompt = query_prompt.replace('[COLUMNS]', f",".join(columns))
    
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    parsed_response = LLM_parse_query(args, InfList, query_prompt, message)
    tasks_list, descs_list, key_node_list = parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
    logger.debug(f"Parsed tasks: {len(tasks_list)}, descriptions: {len(descs_list)}, key nodes: {len(key_node_list)}", "Demo")
    #tasks_list, descs_list, key_node_list = ['Treatment Effect Estimation'], ['Analyze the treatment effect of PIP2 to PIP3.'], ['PIP3']
    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
        logger.status("Task", task)
        logger.status("Description", desc[:50] + "..." if len(desc) > 50 else desc)
        logger.status("Key node", key_node)
        response, figs = analysis.forward(task, desc, key_node)
        logger.section("Result Analysis")
        logger.success("Analysis completed")
        for file_name in figs:
            img = mpimg.imread(f'{file_name}')  # Read the image
            plt.imshow(img)  
            plt.show()
    
    
if __name__ == '__main__':
    import argparse
    import pickle
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="demo_data/20250121_223113/lalonde/lalonde.csv",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='causal_inference/test_result',
            help='Directory to save the output report'
        )

        # Output directory for graphs
        parser.add_argument(
            '--output-graph-dir',
            type=str,
            default='causal_inference/test_result',
            help='Directory to save the output graph'
        )

        # OpenAI Settings
        parser.add_argument(
            '--organization',
            type=str,
            default="org-gw7mBMydjDsOnDlTvNQWXqPL",
            help='Organization ID'
        )

        parser.add_argument(
            '--project',
            type=str,
            default="proj_SIDtemBJMHUWG7CPdU7yRjsn",
            help='Project ID'
        )

        parser.add_argument(
            '--apikey',
            type=str,
            default=None,
            help='API Key'
        )

        parser.add_argument(
            '--simulation_mode',
            type=str,
            default="offline",
            help='Simulation mode: online or offline'
        )

        parser.add_argument(
            '--data_mode',
            type=str,
            default="real",
            help='Data mode: real or simulated'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help='Enable debugging mode'
        )

        parser.add_argument(
            '--initial_query',
            type=str,
            default="selected algorithm: PC",
            help='Initial query for the algorithm'
        )
        args = parser.parse_args()
        return args
    
    args = parse_args()
    with open('demo_data/20250121_223113/lalonde/output_graph/PC_global_state.pkl', 'rb') as file:
        global_state = pickle.load(file)

    main(global_state, args)
