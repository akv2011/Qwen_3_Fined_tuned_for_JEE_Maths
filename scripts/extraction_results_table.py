"""
PDF Extraction Results Table Generator
Creates a comprehensive comparison table of all extraction methods
"""

import pandas as pd
from tabulate import tabulate

def create_extraction_results_table():
    """Create a comprehensive table of extraction results"""
    
    # Data from the comparison results
    results_data = [
        {
            'Library': 'LlamaParse',
            'Status': '✓ Success',
            'Processing Time (s)': 12.05,
            'Text Length (chars)': '1,837,506',
            'Elements/Pages': 626,
            'Math Formulas': 47291,
            'Problems Detected': 8587,
            'Speed Rank': 2,
            'Content Rank': 1,
            'Best For': 'Content Quality & Math Problem Extraction',
            'API Required': 'Llama Cloud',
            'Notes': 'Clear winner with 32x more problems than baseline'
        },
        {
            'Library': 'PyMuPDF',
            'Status': '✓ Success', 
            'Processing Time (s)': 6.13,
            'Text Length (chars)': '1,772,850',
            'Elements/Pages': 625,
            'Math Formulas': 45920,
            'Problems Detected': 263,
            'Speed Rank': 1,
            'Content Rank': 3,
            'Best For': 'Speed & Baseline Extraction',
            'API Required': 'None',
            'Notes': 'Fastest processing, good baseline performance'
        },
        {
            'Library': 'Unstructured',
            'Status': '✓ Success',
            'Processing Time (s)': 1881.20,
            'Text Length (chars)': '1,349,507',
            'Elements/Pages': 25308,
            'Math Formulas': 28612,
            'Problems Detected': 269,
            'Speed Rank': 3,
            'Content Rank': 2,
            'Best For': 'Detailed Layout Analysis',
            'API Required': 'None (Tesseract OCR)',
            'Notes': 'Most detailed structure but very slow (31+ minutes)'
        },
        {
            'Library': 'LangExtract',
            'Status': '✗ Failed',
            'Processing Time (s)': 'N/A',
            'Text Length (chars)': 'N/A',
            'Elements/Pages': 'N/A',
            'Math Formulas': 'N/A',
            'Problems Detected': 'N/A',
            'Speed Rank': 'N/A',
            'Content Rank': 'N/A',
            'Best For': 'Structured Grounded Extraction',
            'API Required': 'Google Gemini',
            'Notes': 'Failed due to API quota limits (10 requests/min free tier)'
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    return df

def print_summary_table():
    """Print a summary table with key metrics"""
    
    print("🏆 PDF EXTRACTION COMPARISON RESULTS")
    print("=" * 80)
    
    # Summary metrics table
    summary_data = [
        ['Library', 'Status', 'Time (s)', 'Problems', 'Math Formulas', 'Recommendation'],
        ['LlamaParse', '✓ Success', '12.05', '8,587', '47,291', '🥇 WINNER - Best Content'],
        ['PyMuPDF', '✓ Success', '6.13', '263', '45,920', '🥈 SPEED - Fastest Processing'],
        ['Unstructured', '✓ Success', '1,881.20', '269', '28,612', '🥉 STRUCTURE - Detailed Layout'],
        ['LangExtract', '✗ Failed', 'N/A', 'N/A', 'N/A', '⚠️ QUOTA - API Limit Exceeded']
    ]
    
    print(tabulate(summary_data, headers='firstrow', tablefmt='grid'))
    
    print("\n📊 PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    # Performance comparison
    performance_data = [
        ['Metric', 'Winner', 'Value', 'Runner-up', 'Value'],
        ['Speed', 'PyMuPDF', '6.13s', 'LlamaParse', '12.05s'],
        ['Problems Found', 'LlamaParse', '8,587', 'Unstructured', '269'],
        ['Math Formulas', 'LlamaParse', '47,291', 'PyMuPDF', '45,920'],
        ['Content Quality', 'LlamaParse', '32x baseline', 'Unstructured', '1.02x baseline'],
        ['Text Length', 'LlamaParse', '1.84M chars', 'PyMuPDF', '1.77M chars']
    ]
    
    print(tabulate(performance_data, headers='firstrow', tablefmt='grid'))
    
    print("\n🎯 RECOMMENDATIONS BY USE CASE")
    print("-" * 40)
    
    recommendations = [
        ['Use Case', 'Recommended Library', 'Reason'],
        ['JEE Dataset Creation', 'LlamaParse', 'Extracts 32x more problems than baseline'],
        ['Quick Text Extraction', 'PyMuPDF', 'Fastest processing at 6.13s'],
        ['Document Structure Analysis', 'Unstructured', 'Most detailed layout with 25K+ elements'],
        ['Structured Data Extraction', 'LangExtract*', '*Requires paid API to avoid quota limits'],
        ['Production Pipeline', 'LlamaParse', 'Best balance of quality and speed'],
        ['Batch Processing', 'PyMuPDF', 'No API dependencies, consistent performance']
    ]
    
    print(tabulate(recommendations, headers='firstrow', tablefmt='grid'))
    
    print("\n💡 KEY INSIGHTS")
    print("-" * 20)
    print("• LlamaParse extracted 8,587 problems vs PyMuPDF's 263 (32x improvement)")
    print("• Speed difference: LlamaParse only 2x slower than PyMuPDF but 32x better content")
    print("• Unstructured provides most detailed structure but 300x slower than PyMuPDF")
    print("• LangExtract has potential for structured extraction but needs paid API")
    print("• For Aryabhata JEE dataset: LlamaParse is the clear choice")

def print_detailed_table():
    """Print detailed comparison table"""
    
    df = create_extraction_results_table()
    
    print("\n📋 DETAILED COMPARISON TABLE")
    print("=" * 100)
    
    # Select key columns for display
    display_columns = [
        'Library', 'Status', 'Processing Time (s)', 
        'Problems Detected', 'Math Formulas', 'Best For', 'Notes'
    ]
    
    display_df = df[display_columns]
    
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

if __name__ == "__main__":
    try:
        print_summary_table()
        print_detailed_table()
        
        print("\n🚀 NEXT STEPS FOR ARYABHATA PROJECT")
        print("-" * 40)
        print("1. Implement LlamaParse production pipeline")
        print("2. Process full JEE mathematics PDF collection")
        print("3. Structure extracted content for training data")
        print("4. Consider LangExtract with paid API for enhanced structure")
        print("5. Aim for >95% JEE Main accuracy with structured dataset")
        
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("pip install pandas tabulate")
        print(f"Error: {e}")
