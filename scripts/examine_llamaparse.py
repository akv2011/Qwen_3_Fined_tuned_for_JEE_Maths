import json

# Load the extraction data and focus on LlamaParse results
print("Examining LlamaParse structured data...")
with open('extraction_results/results_data_20250814_022148.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find LlamaParse entry
llamaparse_entry = None
for entry in data:
    if entry.get('library') == 'LlamaParse':
        llamaparse_entry = entry
        break

if not llamaparse_entry:
    print("LlamaParse entry not found!")
    exit()

print(f"LlamaParse Results:")
print(f"  Success: {llamaparse_entry['success']}")
print(f"  Problems detected: {llamaparse_entry['problems_detected']:,}")
print(f"  Math formulas detected: {llamaparse_entry['math_formulas_detected']:,}")

structured = llamaparse_entry['structured_data']
print(f"\nStructured data contains: {list(structured.keys())}")

if 'documents' in structured:
    documents = structured['documents']
    print(f"Number of documents: {len(documents)}")
    
    if documents:
        print(f"\nFirst document structure:")
        first_doc = documents[0]
        print(f"  Keys: {list(first_doc.keys())}")
        
        # Check if we have text content
        if 'text' in first_doc:
            text_sample = first_doc['text'][:500]
            print(f"  Text sample (first 500 chars): {text_sample}")
        
        # Check for other content
        for key in first_doc.keys():
            if key != 'text':
                print(f"  {key}: {type(first_doc[key])}")
                if isinstance(first_doc[key], list) and first_doc[key]:
                    print(f"    Sample item: {first_doc[key][0] if first_doc[key] else 'Empty'}")
