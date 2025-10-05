#!/usr/bin/env python3
"""
Create HuggingFace Dataset from JEE Mathematics Extraction Results

This script processes the LlamaParse extraction results and creates a
HuggingFace dataset suitable for fine-tuning mathematical models.
"""

import json
import os
import re
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

def extract_mathematical_problems(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract mathematical problems from LlamaParse extraction results.
    
    Args:
        json_file_path: Path to the extraction results JSON file
        
    Returns:
        List of mathematical problems formatted for training
    """
    print(f"Loading extraction data from: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find the LlamaParse entry (best extraction results)
    llamaparse_entry = None
    for entry in data:
        if entry.get('library') == 'LlamaParse' and entry.get('success'):
            llamaparse_entry = entry
            break
    
    if not llamaparse_entry:
        print("❌ No successful LlamaParse entry found!")
        return []
    
    print(f"✅ Found LlamaParse entry:")
    print(f"   📊 Problems detected: {llamaparse_entry.get('problems_detected', 0):,}")
    print(f"   📊 Math formulas detected: {llamaparse_entry.get('math_formulas_detected', 0):,}")
    print(f"   ⏱️  Processing time: {llamaparse_entry.get('processing_time', 0):.1f}s")
    
    structured_data = llamaparse_entry.get('structured_data', {})
    documents = structured_data.get('documents', [])
    
    print(f"📄 Processing {len(documents)} documents...")
    
    problems = []
    
    # Mathematical content indicators
    math_indicators = [
        # Basic math terms
        'solve', 'find', 'calculate', 'prove', 'equation', 'function', 'solution',
        'answer', 'result', 'value', 'expression', 'simplify', 'evaluate',
        
        # Advanced math terms
        'derivative', 'integral', 'limit', 'theorem', 'formula', 'graph',
        'triangle', 'circle', 'polynomial', 'matrix', 'vector', 'angle',
        'probability', 'statistics', 'permutation', 'combination',
        
        # Math symbols and functions (common in text)
        'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'sum', 'product',
        
        # Mathematical operators/symbols that might appear as text
        'integral', 'sigma', 'pi', 'theta', 'alpha', 'beta', 'gamma',
        'infinity', 'degrees', 'radians'
    ]
    
    # Math symbol patterns
    math_symbols = re.compile(r'[∫∑√±≤≥≠∈∉∪∩∆∇∂φψωαβγδεζηθικλμνξοπρστυχψω]')
    
    for i, doc in enumerate(documents):
        text = doc.get('text', '').strip()
        document_id = doc.get('document_id', i)
        metadata = doc.get('metadata', {})
        
        # Skip very short texts or empty documents
        if len(text) < 50:
            continue
        
        # Check for mathematical content
        text_lower = text.lower()
        has_math_terms = any(indicator in text_lower for indicator in math_indicators)
        has_math_symbols = bool(math_symbols.search(text))
        has_numbers = bool(re.search(r'\d+', text))
        
        # Filter for mathematical content
        if has_math_terms or has_math_symbols:
            # Create training example in conversational format
            problem = {
                'id': f"jee_math_doc_{document_id}",
                'conversations': [
                    {
                        'role': 'user',
                        'content': f"Solve this JEE mathematics problem:\n\n{text}"
                    },
                    {
                        'role': 'assistant', 
                        'content': "I'll solve this step by step.\n\n[This would contain the detailed solution - currently extracted text contains the problem statement]"
                    }
                ],
                'source': 'JEE Mathematics - 41 Years IIT JEE by Amit M Agarwal',
                'document_id': document_id,
                'extraction_method': 'LlamaParse',
                'text_length': len(text),
                'has_math_symbols': has_math_symbols,
                'has_math_terms': has_math_terms,
                'original_text': text,
                'metadata': metadata
            }
            problems.append(problem)
    
    print(f"✅ Extracted {len(problems)} mathematical problems")
    return problems

def create_huggingface_dataset(problems: List[Dict[str, Any]], output_dir: str = "jee_math_dataset"):
    """
    Create HuggingFace dataset from extracted problems.
    
    Args:
        problems: List of mathematical problems
        output_dir: Directory to save the dataset
    """
    if not problems:
        print("❌ No problems to create dataset from!")
        return None
    
    print(f"📦 Creating HuggingFace dataset with {len(problems)} examples...")
    
    # Create dataset
    dataset = Dataset.from_list(problems)
    
    # Split into train/test (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })
    
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    print(f"✅ Dataset saved to: {output_dir}")
    print(f"   📊 Training examples: {len(dataset_dict['train'])}")
    print(f"   📊 Test examples: {len(dataset_dict['test'])}")
    
    # Also save as JSON for easy inspection
    json_output = os.path.join(output_dir, "dataset.json")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Also saved as JSON: {json_output}")
    
    return dataset_dict

def main():
    """Main execution function"""
    print("🚀 JEE Mathematics Dataset Creator")
    print("=" * 50)
    
    # Find the latest extraction results
    results_dir = "extraction_results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for the largest/most recent results file
    results_files = [f for f in os.listdir(results_dir) if f.startswith('results_data_') and f.endswith('.json')]
    
    if not results_files:
        print(f"❌ No results files found in {results_dir}")
        return
    
    # Use the most recent file (or you can specify manually)
    latest_file = sorted(results_files)[-1]  # Last alphabetically = most recent
    json_file_path = os.path.join(results_dir, latest_file)
    
    print(f"📁 Using extraction file: {json_file_path}")
    print(f"📊 File size: {os.path.getsize(json_file_path) / (1024*1024):.1f} MB")
    print()
    
    # Extract problems
    problems = extract_mathematical_problems(json_file_path)
    
    if not problems:
        print("❌ No mathematical problems extracted!")
        return
    
    # Create dataset
    dataset = create_huggingface_dataset(problems)
    
    if dataset:
        print("\n🎉 Dataset creation completed successfully!")
        print("\n📋 Dataset Summary:")
        print(f"   • Total examples: {len(problems)}")
        print(f"   • Training examples: {len(dataset['train'])}")
        print(f"   • Test examples: {len(dataset['test'])}")
        print(f"   • Average text length: {sum(p['text_length'] for p in problems) / len(problems):.0f} characters")
        
        # Show a sample
        print("\n📝 Sample training example:")
        sample = problems[0]
        print(f"   ID: {sample['id']}")
        print(f"   Text preview: {sample['original_text'][:200]}...")
        print(f"   Has math symbols: {sample['has_math_symbols']}")
        print(f"   Has math terms: {sample['has_math_terms']}")

if __name__ == "__main__":
    main()
