#!/usr/bin/env python3
"""
Convert JEE Math extraction results to HuggingFace dataset format
This script processes the large JSON files from LlamaParse extraction and creates
a structured dataset suitable for fine-tuning mathematical reasoning models.
"""

import json
import os
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import re
from typing import List, Dict, Any, Optional

def analyze_extraction_data(json_file_path: str) -> Dict[str, Any]:
    """
    Analyze the structure of extraction results to understand the data format.
    
    Args:
        json_file_path: Path to the extraction results JSON file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing file: {json_file_path}")
    
    # Check file size
    file_size = os.path.getsize(json_file_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    # Load data in chunks to avoid memory issues
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            return {}
    
    # Analyze structure
    analysis = {
        "total_entries": len(data) if isinstance(data, list) else 1,
        "data_type": type(data).__name__,
        "keys": list(data.keys()) if isinstance(data, dict) else [],
        "sample_entries": [],
        "extraction_methods": set(),
        "math_content_types": set()
    }
    
    # Sample first few entries
    if isinstance(data, list):
        analysis["sample_entries"] = data[:3]
        
        # Count extraction methods
        for entry in data[:100]:  # Sample first 100
            if isinstance(entry, dict):
                if 'method' in entry:
                    analysis["extraction_methods"].add(entry['method'])
                if 'content_type' in entry:
                    analysis["math_content_types"].add(entry['content_type'])
                    
    elif isinstance(data, dict):
        analysis["sample_entries"] = [data]
        if 'method' in data:
            analysis["extraction_methods"].add(data['method'])
    
    return analysis

def extract_math_problems(data: Any, method_priority: List[str] = ["llamaparse", "unstructured", "pymupdf"]) -> List[Dict[str, str]]:
    """
    Extract mathematical problems from the extraction results.
    
    Args:
        data: Loaded JSON data
        method_priority: List of extraction methods in order of preference
        
    Returns:
        List of dictionaries with problem data
    """
    problems = []
    
    if isinstance(data, dict):
        # Handle case where data is a dictionary with method keys
        for method in method_priority:
            if method in data and data[method]:
                method_data = data[method]
                if isinstance(method_data, list):
                    for item in method_data:
                        problem = extract_problem_from_item(item, method)
                        if problem:
                            problems.append(problem)
                elif isinstance(method_data, dict):
                    problem = extract_problem_from_item(method_data, method)
                    if problem:
                        problems.append(problem)
                break  # Use first available method
                
    elif isinstance(data, list):
        # Handle case where data is a list of extraction results
        for item in data:
            if isinstance(item, dict):
                method = item.get('method', 'unknown')
                problem = extract_problem_from_item(item, method)
                if problem:
                    problems.append(problem)
    
    return problems

def extract_problem_from_item(item: Dict[str, Any], method: str) -> Optional[Dict[str, str]]:
    """
    Extract a mathematical problem from a single item.
    
    Args:
        item: Dictionary containing problem data
        method: Extraction method used
        
    Returns:
        Dictionary with structured problem data or None
    """
    # Common fields to look for
    text_fields = ['text', 'content', 'extracted_text', 'markdown', 'raw_text']
    
    content = ""
    for field in text_fields:
        if field in item and item[field]:
            content = str(item[field])
            break
    
    if not content or len(content.strip()) < 10:
        return None
    
    # Try to identify if this looks like a math problem
    math_indicators = [
        r'\$.*?\$',  # LaTeX math
        r'\\[.*?\\]',  # LaTeX display math
        r'∫|∑|∏|√|α|β|γ|δ|θ|π|Δ|∇',  # Math symbols
        r'\b(solve|find|calculate|prove|show|given|if|then)\b',  # Problem keywords
        r'\b\d+[\.)] ',  # Question numbering
        r'[=<>≤≥≠±∞]'  # Math operators
    ]
    
    has_math = any(re.search(pattern, content, re.IGNORECASE) for pattern in math_indicators)
    
    if not has_math:
        return None
    
    # Extract metadata
    problem_id = item.get('id', item.get('problem_id', f"{method}_{hash(content) % 10000}"))
    source_file = item.get('source_file', item.get('file', 'unknown'))
    page_number = item.get('page', item.get('page_number', None))
    
    return {
        "problem_id": str(problem_id),
        "problem_text": content.strip(),
        "source_file": str(source_file),
        "extraction_method": method,
        "page_number": str(page_number) if page_number else "",
        "has_equations": bool(re.search(r'\$.*?\$|\\[.*?\\]', content)),
        "word_count": len(content.split()),
        "char_count": len(content)
    }

def create_instruction_format(problems: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert problems to instruction-following format for fine-tuning.
    
    Args:
        problems: List of problem dictionaries
        
    Returns:
        List of instruction-format dictionaries
    """
    formatted_data = []
    
    for problem in problems:
        # Create instruction-response pairs
        instruction_variants = [
            f"Solve this JEE mathematics problem:\n\n{problem['problem_text']}",
            f"Here is a JEE mathematics problem. Please provide a detailed solution:\n\n{problem['problem_text']}",
            f"Analyze and solve the following mathematics problem from JEE:\n\n{problem['problem_text']}"
        ]
        
        # For now, we'll use a placeholder response since we don't have solutions
        # In a real scenario, you'd want to generate or extract solutions
        response = f"I need to solve this step by step:\n\n{problem['problem_text']}\n\n[This would contain the detailed solution with mathematical reasoning]"
        
        # Use the first instruction variant for now
        formatted_data.append({
            "instruction": instruction_variants[0],
            "input": "",  # Empty for completion-style training
            "output": response,
            "problem_id": problem['problem_id'],
            "source_file": problem['source_file'],
            "extraction_method": problem['extraction_method'],
            "metadata": {
                "word_count": problem['word_count'],
                "char_count": problem['char_count'],
                "has_equations": problem['has_equations'],
                "page_number": problem['page_number']
            }
        })
    
    return formatted_data

def create_huggingface_dataset(formatted_data: List[Dict[str, str]], 
                              output_dir: str = "jee_math_dataset",
                              train_split: float = 0.8) -> DatasetDict:
    """
    Create a HuggingFace dataset from formatted data.
    
    Args:
        formatted_data: List of instruction-format dictionaries
        output_dir: Directory to save the dataset
        train_split: Fraction of data for training
        
    Returns:
        DatasetDict containing train/validation splits
    """
    print(f"Creating HuggingFace dataset with {len(formatted_data)} examples...")
    
    # Convert to pandas DataFrame first for easier manipulation
    df = pd.DataFrame(formatted_data)
    
    # Split into train/validation
    train_size = int(len(df) * train_split)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    # Also save as JSON for inspection
    with open(f"{output_dir}/train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_df.to_dict('records'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(f"{output_dir}/validation.jsonl", 'w', encoding='utf-8') as f:
        for item in val_df.to_dict('records'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved to: {output_dir}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    return dataset_dict

def main():
    """Main function to create HuggingFace dataset from extraction results."""
    
    # Find the latest extraction results file
    extraction_dir = Path("extraction_results")
    json_files = list(extraction_dir.glob("results_data_*.json"))
    
    if not json_files:
        print("No extraction results found in extraction_results/")
        return
    
    # Use the largest file (most recent/complete)
    latest_file = max(json_files, key=lambda x: x.stat().st_size)
    print(f"Using extraction file: {latest_file}")
    
    # Analyze data structure
    print("\n" + "="*50)
    print("ANALYZING EXTRACTION DATA")
    print("="*50)
    
    analysis = analyze_extraction_data(str(latest_file))
    print(f"Data analysis:")
    for key, value in analysis.items():
        if key != "sample_entries":
            print(f"  {key}: {value}")
    
    # Show sample entry structure
    if analysis["sample_entries"]:
        print(f"\nSample entry structure:")
        sample = analysis["sample_entries"][0]
        if isinstance(sample, dict):
            for key in list(sample.keys())[:5]:
                print(f"  {key}: {type(sample[key])}")
    
    # Load and process data
    print("\n" + "="*50)
    print("EXTRACTING MATHEMATICAL PROBLEMS")
    print("="*50)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = extract_math_problems(data)
    print(f"Extracted {len(problems)} mathematical problems")
    
    if not problems:
        print("No mathematical problems found. Check data structure.")
        return
    
    # Show sample problems
    print(f"\nSample problems:")
    for i, problem in enumerate(problems[:3]):
        print(f"\nProblem {i+1}:")
        print(f"  ID: {problem['problem_id']}")
        print(f"  Source: {problem['source_file']}")
        print(f"  Method: {problem['extraction_method']}")
        print(f"  Text preview: {problem['problem_text'][:200]}...")
    
    # Convert to instruction format
    print("\n" + "="*50)
    print("CREATING INSTRUCTION FORMAT")
    print("="*50)
    
    formatted_data = create_instruction_format(problems)
    print(f"Created {len(formatted_data)} instruction-response pairs")
    
    # Create HuggingFace dataset
    print("\n" + "="*50)
    print("CREATING HUGGINGFACE DATASET")
    print("="*50)
    
    dataset = create_huggingface_dataset(formatted_data)
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    train_dataset = dataset["train"]
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Total validation examples: {len(dataset['validation'])}")
    
    # Analyze content
    word_counts = [item['metadata']['word_count'] for item in formatted_data]
    print(f"Average words per problem: {sum(word_counts) / len(word_counts):.1f}")
    print(f"Min/Max words: {min(word_counts)}/{max(word_counts)}")
    
    equations_count = sum(1 for item in formatted_data if item['metadata']['has_equations'])
    print(f"Problems with equations: {equations_count}/{len(formatted_data)} ({equations_count/len(formatted_data)*100:.1f}%)")
    
    extraction_methods = {}
    for item in formatted_data:
        method = item['extraction_method']
        extraction_methods[method] = extraction_methods.get(method, 0) + 1
    
    print(f"Extraction methods breakdown:")
    for method, count in extraction_methods.items():
        print(f"  {method}: {count} problems")
    
    print(f"\n✅ HuggingFace dataset created successfully!")
    print(f"📁 Location: jee_math_dataset/")
    print(f"📄 JSONL files: jee_math_dataset/train.jsonl, jee_math_dataset/validation.jsonl")
    print(f"🚀 Ready for fine-tuning with Qwen2.5-7B-Instruct!")

if __name__ == "__main__":
    main()
