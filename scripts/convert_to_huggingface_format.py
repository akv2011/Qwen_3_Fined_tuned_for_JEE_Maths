#!/usr/bin/env python3
"""
Convert JEE Math Extraction Results to HuggingFace Dataset Format
Transforms LlamaParse extraction results into instruction-response format for fine-tuning
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

def clean_math_text(text: str) -> str:
    """Clean and normalize mathematical text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common math notation issues
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'\\t', ' ', text)
    
    return text

def extract_problems_from_llamaparse(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract individual problems from LlamaParse results"""
    problems = []
    
    # Look for structured problem patterns in the text
    if 'text' in data:
        text = data['text']
        
        # Split by common problem indicators
        problem_patterns = [
            r'(?:Problem|Question|Q\.?)\s*\d+',
            r'(?:Example|Ex\.?)\s*\d+',
            r'\d+\.\s*',
            r'(?:Find|Calculate|Solve|Determine)',
        ]
        
        # Split text into potential problems
        for pattern in problem_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = match.start()
                # Find next problem or end of text
                next_match = re.search(pattern, text[start + 1:], re.IGNORECASE)
                if next_match:
                    end = start + 1 + next_match.start()
                else:
                    end = min(start + 1000, len(text))  # Limit problem length
                
                problem_text = text[start:end].strip()
                
                if len(problem_text) > 50:  # Filter out too short segments
                    problems.append({
                        'problem': clean_math_text(problem_text),
                        'source': 'llamaparse_extraction'
                    })
    
    return problems

def convert_to_instruction_format(problems: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert problems to instruction-response format for fine-tuning"""
    conversations = []
    
    for i, problem in enumerate(problems):
        problem_text = problem['problem']
        
        # Try to separate question and solution if both exist
        if 'solution:' in problem_text.lower() or 'answer:' in problem_text.lower():
            # Split at solution/answer marker
            parts = re.split(r'(?:solution|answer):\s*', problem_text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                question = parts[0].strip()
                solution = 'Solution: ' + parts[1].strip()
            else:
                question = problem_text
                solution = "I need to solve this step by step."
        else:
            question = problem_text
            solution = "I need to solve this step by step."
        
        # Create instruction-response pair
        conversation = {
            "id": f"jee_math_{i+1:06d}",
            "instruction": f"Solve this JEE Mathematics problem:\n\n{question}",
            "response": solution,
            "source": problem.get('source', 'unknown'),
            "subject": "mathematics",
            "level": "jee_advanced"
        }
        
        conversations.append(conversation)
    
    return conversations

def main():
    parser = argparse.ArgumentParser(description='Convert JEE extraction results to HuggingFace format')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file from extraction results')
    parser.add_argument('--output', '-o', default='jee_math_huggingface_dataset.jsonl', help='Output JSONL file')
    parser.add_argument('--max-problems', '-m', type=int, default=10000, help='Maximum number of problems to process')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"🔄 Loading extraction results from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {input_path.stat().st_size / (1024*1024):.1f}MB of data")
        
        # Extract problems from LlamaParse format
        print("🔍 Extracting problems from LlamaParse results...")
        problems = extract_problems_from_llamaparse(data)
        print(f"📊 Found {len(problems)} potential problems")
        
        # Limit number of problems if specified
        if len(problems) > args.max_problems:
            problems = problems[:args.max_problems]
            print(f"📝 Limited to {args.max_problems} problems")
        
        # Convert to instruction format
        print("🔄 Converting to instruction-response format...")
        conversations = convert_to_instruction_format(problems)
        
        # Save as JSONL
        print(f"💾 Saving to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully created dataset with {len(conversations)} examples")
        print(f"📁 Output file: {output_path} ({output_path.stat().st_size / (1024*1024):.1f}MB)")
        
        # Create summary
        summary = {
            "total_examples": len(conversations),
            "source_file": str(input_path),
            "created_at": datetime.now().isoformat(),
            "format": "instruction_response",
            "target_model": "Qwen/Qwen2.5-7B-Instruct"
        }
        
        summary_path = output_path.with_suffix('.summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
