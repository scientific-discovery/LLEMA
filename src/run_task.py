#!/usr/bin/env python3
"""
General task runner for all LLEMA tasks
Takes dataset argument to specify which task to run
"""
import os
import sys
import argparse
import csv

# Add the agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))

def get_task_info(dataset_name):
    """Get task information from consolidated CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), 'all_tasks.csv')
    
    # Normalize the input dataset name
    normalized_input = dataset_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace(',', '')
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Normalize the task name from CSV
            normalized_task = row['Task'].lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace(',', '')
            if normalized_task == normalized_input:
                return row
    
    # If not found, try to find by partial match
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            normalized_task = row['Task'].lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace(',', '')
            if normalized_input in normalized_task or normalized_task in normalized_input:
                return row
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run LLEMA task with specified dataset')
    parser.add_argument('dataset', help='Dataset/task name to run (e.g., transparent_conductors, photovoltaic_absorbers)')
    parser.add_argument('--no-island-mode', action='store_true', help='Enable no island mode for logging')
    
    args = parser.parse_args()
    
    # Get task information
    task_info = get_task_info(args.dataset)
    if not task_info:
        print(f"Error: Task '{args.dataset}' not found in all_tasks.csv.")
        print("Available tasks:")
        csv_path = os.path.join(os.path.dirname(__file__), 'all_tasks.csv')
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                normalized_name = row['Task'].lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace(',', '')
                print(f"  - {normalized_name}")
        sys.exit(1)
    
    # Set environment variables
    os.environ['TASK_DESCRIPTION_PATH'] = os.path.join(os.path.dirname(__file__), 'all_tasks.csv')
    if args.no_island_mode:
        os.environ['NO_ISLAND_MODE'] = 'true'
    
    print("="*60)
    print(f"Running {task_info['Task']} Task")
    print(f"Goal: {task_info['Goal']}")
    print(f"Properties and Models: {task_info['Properties and Models']}")
    print("="*60)
    
    # Import and run the main function from the agent module
    from main import main as agent_main
    agent_main(task_name=task_info['Task'])

if __name__ == "__main__":
    main()
