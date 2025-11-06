from llm import LLMClient
from utils.logger import logger
class Discussion(object):
    # Kun Zhou Implemented
    def __init__(self, args, report):
        self.args = args
        self.client = LLMClient(args)

        # Extract the text information from the Latex file
        response = self.client.chat_completion(
            prompt=report,
            system_prompt="You are a helpful assistant for extracting text from Latex.",
            json_response=False
        )
        self.report_content = response

    def interaction(self, conversation_history, user_query):
        conversation_history.append({"role": "user", "content": user_query})
        
        try:
            response = self.client.chat_completion(
                prompt=user_query,
                system_prompt="You are a helpful assistant for user interaction.",
                json_response=False
            )
            output = response
            # Format the response for better readability
            logger.info("🤖 Copilot Response:")
            logger.status("", output)  # Empty message to just show the content
            logger.debug("=" * 95, "Discussion")
            return conversation_history, output
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", "Discussion")
            error_output = "I apologize, but I encountered an error while processing your question. Please try rephrasing your question or contact support if the issue persists."
            logger.info("🤖 Copilot Response (Error):")
            logger.status("", error_output)  # Empty message to just show the content
            return conversation_history, error_output

    def forward(self, global_state):
        '''
        :param global_state: The global state containing the processed data, algorithm candidates, statistics description, and knowledge document
        :param report: The string containing the content of the latex file
        '''
        logger.process("Starting interactive discussion session")
        conversation_history = [{"role": "system", "content": "You are a helpful assistant. Please always refer to the following Causal Analysis information to discuss with the user and answer the user's question\n\n%s"%self.report_content}]

        # Answer User Query based on Previous Info
        logger.info("Interactive Q&A session started. Type 'No' to exit.")
        while True:
            logger.info("If you still have any questions, just say it and let me help you! If not, just say No")
            user_query = input("User: ")
            
            if user_query.lower() == "no":
                logger.success("Thank you for using Causal-Copilot! See you!")
                break
                
            conversation_history, output = self.interaction(conversation_history, user_query)
            conversation_history.append({"role": "system", "content": output})

            # Log the conversation to global state
            global_state.logging.final_discuss.append({
                "input": user_query,
                "output": output
            })
        
        logger.success("Discussion session completed successfully")



if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="data/dataset/Abalone/Abalone.csv",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='data/dataset/Abalone/output_report',
            help='Directory to save the output report'
        )

        # Output directory for graphs
        parser.add_argument(
            '--output-graph-dir',
            type=str,
            default='data/dataset/Abalone/output_graph',
            help='Directory to save the output graph'
        )

        # OpenAI Settings
        parser.add_argument(
            '--organization',
            type=str,
            default="org-5NION61XDUXh0ib0JZpcppqS",
            help='Organization ID'
        )

        parser.add_argument(
            '--project',
            type=str,
            default="proj_Ry1rvoznXAMj8R2bujIIkhQN",
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

        parser.add_argument(
            '--parallel',
            type=bool,
            default=False,
            help='Parallel computing for bootstrapping.'
        )

        parser.add_argument(
            '--demo_mode',
            type=bool,
            default=False,
            help='Demo mode'
        )

        args = parser.parse_args()
        return args
    args = parse_args()

    logger.process("Running Discussion module in standalone mode")
    try:
        report = open("../output/Abalone.csv/20241202_205252/output_report/report.txt").read()
        logger.success("Report file loaded successfully")
        
        discussion = Discussion(args, report)
        
        # Create a mock global_state for testing
        from types import SimpleNamespace
        mock_global_state = SimpleNamespace()
        mock_global_state.logging = SimpleNamespace()
        mock_global_state.logging.final_discuss = []
        
        discussion.forward(mock_global_state)
        
        logger.info(f"Discussion session completed with {len(mock_global_state.logging.final_discuss)} interactions recorded")
        
    except FileNotFoundError as e:
        logger.error(f"Report file not found: {e}")
    except Exception as e:
        logger.error(f"Error during discussion execution: {e}")