"""
Sample test for JEE PDF extraction comparison
"""
import asyncio
from jee_extraction_comparison import JEEExtractionComparison

async def test_comparison():
    # Update this path to your JEE Math PDF
    pdf_path = "your_jee_math_file.pdf"
    
    print(f"Testing PDF extraction with: {pdf_path}")
    
    comparison = JEEExtractionComparison(pdf_path)
    results = await comparison.run_comparison()
    
    print("\n" + comparison.generate_comparison_report())
    comparison.save_results()

if __name__ == "__main__":
    print("JEE PDF Extraction Test")
    print("Make sure to update the pdf_path variable with your actual PDF file!")
    
    # Uncomment the next line to run the test
    # asyncio.run(test_comparison())
