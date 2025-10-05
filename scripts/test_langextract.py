#!/usr/bin/env python3
"""Quick test of LangExtract functionality"""

import os
from jee_extraction_comparison import JEEExtractionComparison

def main():
    # Set API key
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyACHvqkA6UHMcZwSnhSuB50lhrnJzxOAjg'
    
    # Test LangExtract
    comparison = JEEExtractionComparison('41 Years IIT JEE Mathematics by Amit M Agarwal.pdf')
    result = comparison.extract_with_langextract()
    
    print(f'LangExtract Status: {"Success" if result.success else "Failed"}')
    if result.success:
        print(f'Processing Time: {result.processing_time:.2f}s')
        print(f'Problems Detected: {result.problems_detected}')
        print(f'Math Formulas: {result.math_formulas_detected}')
        print(f'Text Length: {result.text_length}')
        print(f'Total Extractions: {result.structured_data.get("total_extractions", 0)}')
    else:
        print(f'Error: {result.error}')

if __name__ == "__main__":
    main()
