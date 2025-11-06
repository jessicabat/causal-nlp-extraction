from llm import LLMClient
import json
import os

# Class to get hte_algo suggested by LLM
class HTE_Filter(object):
    def __init__(self, args):
        self.args = args
        self.client = LLMClient(args)

    def load_algo_context(self):
        return open(f"causal_inference/IV/context/hte_algo.txt", "r").read()
    
    def load_select_prompt(self):
        return open(f"causal_inference/IV/context/hte_select_prompt.txt", "r").read()

    def create_prompt(self, data, statistics_desc, question):
        columns = ', '.join(data.columns)

        algo_context = self.load_algo_context()
        prompt_template = self.load_select_prompt()

        replacements = {
            "[COLUMNS]": columns,
            "[STATISTICS_DESC]": statistics_desc,
            "[ALGO_CONTEXT]": algo_context,
            "[QUESTION]": question
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template 

    def forward(self, global_state, query):
        prompt = self.create_prompt(global_state.user_data.processed_data, global_state.statistics.description, query)

        response = self.client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for IV hte_filter.",
            json_response=True
        )
        hte_algo = response
        from utils.logger import logger
        logger.debug(f"HTE algorithm response received", "HTE")

        global_state.inference.hte_algo_json = hte_algo

        return global_state
