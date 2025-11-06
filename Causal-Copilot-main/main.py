# Suppress external library logs first
import utils.suppress_logs

# Import logger after suppressing external logs
from utils.logger import logger

from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from causal_discovery.filter import Filter
from causal_discovery.program import Programming
from causal_discovery.rerank import Reranker
from causal_discovery.hyperparameter_selector import HyperparameterSelector
from postprocess.judge import Judge
from postprocess.visualization import Visualization, convert_to_edges
from preprocess.eda_generation import EDA
from report.report_generation import Report_generation
from global_setting.Initialize_state import global_state_initialization, load_data

import os
import json
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        # default= "simulated_data/default/data.csv",
        default= "data/dataset/Abalone/Abalone.csv",
        help='Path to the input dataset file (e.g., CSV format or directory location)'
    )

    # Output file for results
    parser.add_argument(
        '--output-report-dir',
        type=str,
        # default='data/dataset/sim_ts/output_report/',
        default='output/Abalone',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        # default='data/dataset/sim_ts/output_graph/',
        default='output/Abalone',
        help='Directory to save the output graph'
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
        default="Do causal discovery on this dataset",
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

def load_real_world_data(file_path):
    #Baseline code
    # Checking file format and loading accordingly, right now it's for CSV only
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
    
    # Show basic dataset information
    data_info = {
        "Shape": f"({data.shape[0]:,} rows, {data.shape[1]} columns)",
        "Columns": f"{list(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}",
        "Memory usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
        "Data types": f"{data.dtypes.value_counts().to_dict()}"
    }
    logger.data_info("Dataset loaded successfully", data_info)
    return data

def process_user_query(query, data):
    logger.detail(f"Processing user query: {query[:100]}...")
    
    #Baseline code
    query_dict = {}
    original_shape = data.shape
    
    if ';' in query or ':' in query:
        for part in query.split(';'):
            if ':' in part:
                key, value = part.strip().split(':')
                query_dict[key.strip()] = value.strip()

    if 'filter' in query_dict and query_dict['filter'] == 'continuous':
        # Filtering continuous columns, just for target practice right now
        data = data.select_dtypes(include=['float64', 'int64'])
        logger.detail(f"Filtered to continuous columns: {original_shape} → {data.shape}")
    
    if 'selected_algorithm' in query_dict:
        selected_algorithm = query_dict['selected_algorithm']
        logger.algorithm("Algorithm manually selected", selected_algorithm)

    # Show query processing results
    processing_results = {
        "Original query": query[:50] + "..." if len(query) > 50 else query,
        "Parsed parameters": len(query_dict),
        "Data shape after processing": f"{data.shape}",
        "Columns selected": f"{len(data.columns)} columns"
    }
    logger.data_info("User query processed", processing_results)
    return data

def main(args):
    logger.header("Causal-Copilot Analysis Session")
    
    logger.step(1, 8, "Initializing global state")
    global_state = global_state_initialization(args)
    logger.detail("Global state initialized successfully")
    
    logger.step(2, 8, "Loading and preparing data")
    global_state = load_data(global_state, args)
    logger.detail("Data loading completed")

    if args.data_mode == 'real':
        global_state.user_data.raw_data = load_real_world_data(args.data_file)
    
    logger.step(3, 8, "Processing user query")
    global_state.user_data.processed_data = process_user_query(args.initial_query, global_state.user_data.raw_data)
    global_state.user_data.visual_selected_features = global_state.user_data.processed_data.columns.tolist()
    global_state.user_data.selected_features = global_state.user_data.processed_data.columns.tolist()

    logger.step(4, 8, "Collecting statistical information")
    if args.debug:
        logger.detail("Using debug mode with fake statistics")
        # Fake statistics for debugging
        global_state.statistics.sample_size = 853
        global_state.statistics.feature_number = 11
        global_state.statistics.missingness = False
        global_state.statistics.data_type = "Continuous"
        global_state.statistics.linearity = True
        global_state.statistics.gaussian_error = True
        global_state.statistics.stationary = "non time-series"
        global_state.user_data.processed_data = global_state.user_data.raw_data
        global_state.user_data.knowledge_docs = "This is fake domain knowledge for debugging purposes."
        logger.detail("Debug statistics and knowledge information loaded")
    else:
        logger.detail("Analyzing dataset characteristics...")
        global_state = stat_info_collection(global_state)
        
        logger.detail("Collecting domain knowledge...")
        global_state = knowledge_info(args, global_state)

    # Convert statistics to text
    global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

    # Show detailed data information
    data_details = {
        "Shape": f"{global_state.user_data.processed_data.shape}",
        "Columns": len(global_state.user_data.processed_data.columns),
        "Missing values": global_state.user_data.processed_data.isnull().sum().sum(),
        "Data type": getattr(global_state.statistics, 'data_type', 'Unknown')
    }
    logger.data_info("Dataset preprocessed", data_details)

    logger.checkpoint("Data preprocessing completed")
    
    #############EDA###################
    logger.step(5, 8, "Exploratory Data Analysis")
    logger.detail("Generating statistical summaries and visualizations...")
    my_eda = EDA(global_state)
    my_eda.generate_eda()
    logger.detail("EDA completed - visualizations saved")
    
    logger.step(6, 8, "Algorithm Selection")
    
    logger.detail("Step 1/3: Filtering suitable algorithms")
    filter = Filter(args)
    global_state = filter.forward(global_state)
    if hasattr(global_state.algorithm, 'filtered_algorithms'):
        logger.detail(f"Filtered to {len(global_state.algorithm.filtered_algorithms)} candidate algorithms")
    else:
        logger.detail("Algorithm filtering completed")

    logger.detail("Step 2/3: Ranking algorithms by suitability")
    reranker = Reranker(args)
    global_state = reranker.forward(global_state)
    logger.detail("Algorithm ranking completed")

    logger.detail("Step 3/3: Optimizing hyperparameters")
    hp_selector = HyperparameterSelector(args)
    global_state = hp_selector.forward(global_state)
    
    logger.algorithm("Selected algorithm", global_state.algorithm.selected_algorithm)
    if hasattr(global_state.algorithm, 'hyperparameters'):
        logger.detail(f"Hyperparameters: {len(global_state.algorithm.hyperparameters)} parameters optimized")
    else:
        logger.detail("Hyperparameter optimization completed")
    
    logger.step(7, 8, "Algorithm Execution")
    logger.detail(f"Running {global_state.algorithm.selected_algorithm} algorithm...")
    try:
        programmer = Programming(args)
        global_state = programmer.forward(global_state)
        logger.detail("Algorithm execution completed")
        
        # Show graph statistics
        if hasattr(global_state.results, 'converted_graph'):
            graph = global_state.results.converted_graph
            if graph is not None:
                edges = (graph != 0).sum()
                logger.detail(f"Discovered {edges} edges in causal graph")
            else:
                logger.warning("No graph result found")
        else:
            logger.warning("No results attribute found")
        
        logger.checkpoint("Causal discovery completed")
    except Exception as e:
        logger.error(f"Algorithm execution failed: {str(e)}")
        raise

    #############Visualization for Initial Graph###################
    my_visual_initial = Visualization(global_state)
    if global_state.statistics.time_series and global_state.results.lagged_graph is not None:
            converted_graph = global_state.results.lagged_graph
            pos_est = my_visual_initial.get_pos(converted_graph[0])
            for i in range(converted_graph.shape[0]):
                _ = my_visual_initial.plot_pdag(converted_graph[i], f'{global_state.algorithm.selected_algorithm}_initial_graph_{i}.svg', pos=pos_est)
            summary_graph = np.any(converted_graph, axis=0).astype(int)
            # pos_est = my_visual_initial.get_pos(summary_graph)
            _ = my_visual_initial.plot_pdag(summary_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph_summary.svg', pos=pos_est)
            my_report = Report_generation(global_state, args)
    else:
        # Get the position of the nodes
        pos_est = my_visual_initial.get_pos(global_state.results.converted_graph)
        # Plot True Graph
        if global_state.user_data.ground_truth is not None:
            _ = my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos=pos_est)
        # Plot Initial Graph
        _ = my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.pdf', pos=pos_est)
        my_report = Report_generation(global_state, args)
        global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
        global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
        judge = Judge(global_state, args)
        if global_state.user_data.ground_truth is not None:
            logger.section("Graph Evaluation")
            logger.detail("Comparing with ground truth graph")
            global_state.results.metrics = judge.evaluation(global_state)
            if hasattr(global_state.results, 'metrics') and global_state.results.metrics:
                logger.metrics_table(global_state.results.metrics, "Performance Metrics")
        
        logger.section("Graph Refinement")
        logger.detail("Applying bootstrap sampling and statistical tests")
        global_state = judge.forward(global_state, 'cot_all_relation', 1)
        logger.success("Graph refinement completed")
    
    #############Visualization for Revised Graph###################
    logger.section("Graph Visualization")
    logger.detail("Generating visualization for revised graph and confidence heatmaps")
    
    # Plot Revised Graph
    my_visual_revise = Visualization(global_state)
    pos_new = my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.pdf', pos=pos_est)
    global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.revised_graph)
    
    # Plot Bootstrap Heatmap - CRITICAL: This must happen before report generation
    logger.detail("Generating bootstrap confidence heatmaps")
    boot_heatmap_paths = my_visual_revise.boot_heatmap_plot()
    if boot_heatmap_paths:
        logger.success(f"Generated {len(boot_heatmap_paths)} confidence heatmap(s)")
        for path in boot_heatmap_paths:
            logger.debug(f"Generated heatmap: {os.path.basename(path)}", "Visualization")
    else:
        logger.warning("No confidence heatmaps were generated (bootstrap data may be empty)")
    
    # global_state.results.refutation_analysis = judge.graph_refutation(global_state)

    # algorithm selection process
    '''
    round = 0
    flag = False

    while round < args.max_iterations and flag == False:
        code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
        flag, algorithm_setup = judge(preprocessed_data, code, results, statistics_dict, algorithm_setup, knowledge_docs)
    '''
    logger.step(8, 8, "Report Generation")
    #############Report Generation###################
    try_num = 1
    
    logger.detail("Step 1/3: Analyzing causal relationships")
    try:
        global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
        global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
        analysis_clean = global_state.logging.graph_conversion['initial_graph_analysis'].replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
        logger.detail("Causal relationship analysis completed")
    except Exception as e:
        logger.error(f"Causal relationship analysis failed: {str(e)}")
        raise
    
    logger.detail("Step 2/3: Generating comprehensive report")
    try:
        my_report = Report_generation(global_state, args)
        report = my_report.generation()
        my_report.save_report(report)
        report_path = os.path.join(global_state.user_data.output_report_dir, 'report.pdf')  
        
        while (not os.path.isfile(report_path)) and try_num<3:
            try_num += 1
            logger.warning(f"Report generation failed, retrying ({try_num}/3)")
            report_gen = Report_generation(global_state, args)
            report = report_gen.generation(debug=False)
            report_gen.save_report(report)
        
        if os.path.isfile(report_path):
            logger.detail("Step 3/3: Report saved successfully")
            logger.detail(f"Report location: {os.path.basename(report_path)}")
            
            # Show final summary
            final_summary = {
                "Algorithm used": global_state.algorithm.selected_algorithm,
                "Graph edges": f"{(global_state.results.converted_graph != 0).sum() if hasattr(global_state.results, 'converted_graph') and global_state.results.converted_graph is not None else 'Unknown'}",
                "Report saved": os.path.basename(report_path),
                "Output directory": os.path.basename(global_state.user_data.output_report_dir)
            }
            logger.data_info("Analysis completed successfully", final_summary)
        else:
            logger.error("Report generation failed after 3 attempts")
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise
    ################################

    # User discussion part
    from user.discuss import Discussion
    discussion = Discussion(args, report)
    discussion.forward(global_state)
    
    logger.checkpoint("Analysis session completed")
    logger.elapsed_time("Total analysis time")

    return report, global_state

if __name__ == '__main__':
    args = parse_args()
    main(args)
            
