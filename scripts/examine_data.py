import json
import os

# Load the extraction data
print("Loading extraction data...")
with open('extraction_results/results_data_20250814_022148.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Data type: {type(data)}")
print(f"Number of entries: {len(data)}")
print("\n" + "="*50)
print("EXTRACTION RESULTS ANALYSIS")
print("="*50)

for i, entry in enumerate(data):
    print(f"\nEntry {i}:")
    print(f"  Library: {entry.get('library', 'Unknown')}")
    print(f"  Success: {entry.get('success', False)}")
    print(f"  Processing time: {entry.get('processing_time', 0):.2f}s")
    print(f"  Text length: {entry.get('text_length', 0):,}")
    print(f"  Math formulas detected: {entry.get('math_formulas_detected', 0):,}")
    print(f"  Problems detected: {entry.get('problems_detected', 0):,}")
    
    structured = entry.get('structured_data', {})
    if structured:
        print(f"  Structured data keys: {list(structured.keys())}")
        if 'problems' in structured:
            problems = structured['problems']
            print(f"    Number of problems: {len(problems)}")
            if problems:
                print(f"    Sample problem keys: {list(problems[0].keys())}")
    else:
        print("  No structured data found")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
