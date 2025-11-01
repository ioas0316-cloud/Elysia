"""
Test script for the full ReAct loop.

This script can now be used in multiple ways:
1.  Without arguments: Runs the first pass (Reasoning).
2.  With `--input <string>`: Takes a string as input (for short observations).
3.  With `--input-file <path>`: Reads input from a file (for long observations).
"""
import json
import argparse
from typing import Dict, Any
from chat_interface import ChatInterface

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input string to process.')
    parser.add_argument('--input-file', type=str, help='Path to a file to use as input.')
    args = parser.parse_args()

    chat = ChatInterface()
    context = {'speaker': 'test_user'}
    user_input = None

    if args.input_file:
        print(f"[Test Script] Reading input from file: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            user_input = f.read()
    elif args.input:
        user_input = args.input
    else:
        # First pass: No input provided, use the default prompt.
        print("--- Running Test: Tool Usage Prompt (First Pass) ---")
        user_input = "Please read the content of the 'available_tools.txt' file for me."
    
    print(f"Input > {user_input[:200]}...")
    output = chat.process_input(user_input, context=context)
    
    if isinstance(output, dict) and "tool_name" in output:
        # A tool call was requested. Print it for the agent to execute.
        print(f"TOOL_CALL::{json.dumps(output)}")
    else:
        # A direct text response was given.
        response, emotion = output
        print(f"Elysia > {response}")

if __name__ == "__main__":
    main()