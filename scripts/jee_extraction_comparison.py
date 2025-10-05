"""
JEE Math PDF Extraction Comparison
Compares LangExtract, LlamaParse, PyMuPDF, and Unstructured libraries
for extracting structured data from JEE Mathematics PDFs.
"""

import os
import time
import json
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Import existing PyMuPDF (already working)
import fitz

# For LangExtract
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("LangExtract not available. Install with: pip install google-langextract")

# For LlamaParse  
try:
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False
    print("LlamaParse not available. Install with: pip install llama-parse llama-index")

# For Unstructured
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Unstructured not available. Install with: pip install unstructured[pdf]")

@dataclass
class ExtractionResult:
    """Results from a PDF extraction method"""
    library: str
    processing_time: float
    success: bool
    raw_content: Any
    structured_data: Dict[str, Any]
    error: Optional[str] = None
    text_length: int = 0
    element_count: int = 0
    math_formulas_detected: int = 0
    problems_detected: int = 0

class JEEExtractionComparison:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.results: List[ExtractionResult] = []
        
        # API keys (set these in environment or config)
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        self.llamaparse_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        
    def extract_with_pymupdf(self) -> ExtractionResult:
        """Extract using PyMuPDF (fitz) - our baseline"""
        print("Testing PyMuPDF extraction...")
        start_time = time.time()
        
        try:
            doc = fitz.open(self.pdf_path)
            full_text = ""
            structured_data = {
                "pages": [],
                "total_pages": len(doc),
                "metadata": doc.metadata
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += page_text + "\n"
                
                # Extract page-level data
                page_data = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "text_length": len(page_text),
                    "blocks": []
                }
                
                # Get text blocks with position info
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        page_data["blocks"].append({
                            "bbox": block["bbox"],
                            "text": " ".join([
                                " ".join([span["text"] for span in line["spans"]])
                                for line in block["lines"]
                            ])
                        })
                
                structured_data["pages"].append(page_data)
            
            doc.close()
            
            # Simple heuristics for math content
            math_indicators = ["=", "∫", "∑", "√", "π", "α", "β", "γ", "θ", "∆", "∇"]
            math_formulas = sum(full_text.count(indicator) for indicator in math_indicators)
            
            # Problem detection (simple heuristic)
            problem_indicators = ["Q.", "Problem", "Find", "Calculate", "Solve", "Prove"]
            problems = sum(full_text.count(indicator) for indicator in problem_indicators)
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                library="PyMuPDF",
                processing_time=processing_time,
                success=True,
                raw_content=full_text,
                structured_data=structured_data,
                text_length=len(full_text),
                element_count=len(structured_data["pages"]),
                math_formulas_detected=math_formulas,
                problems_detected=problems
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                library="PyMuPDF",
                processing_time=processing_time,
                success=False,
                raw_content="",
                structured_data={},
                error=str(e)
            )
    
    def extract_with_langextract(self) -> ExtractionResult:
        """Extract using Google LangExtract with grounded extraction"""
        print("Testing LangExtract extraction...")
        start_time = time.time()
        
        if not LANGEXTRACT_AVAILABLE:
            return ExtractionResult(
                library="LangExtract",
                processing_time=0,
                success=False,
                raw_content="",
                structured_data={},
                error="LangExtract not installed"
            )
        
        if not self.gemini_api_key:
            return ExtractionResult(
                library="LangExtract",
                processing_time=0,
                success=False,
                raw_content="",
                structured_data={},
                error="GOOGLE_API_KEY not set"
            )
        
        try:
            # First read the PDF text
            import fitz
            doc = fitz.open(str(self.pdf_path))
            pdf_text = ""
            for page_num in range(min(10, len(doc))):  # First 10 pages for demo
                pdf_text += doc[page_num].get_text() + "\n"
            doc.close()

            # Define prompt for mathematical content extraction
            prompt_description = """
            Extract JEE mathematics problems with their details.
            Extract problem numbers, question text, mathematical expressions, 
            solution steps, answers, topic classification, and difficulty level.
            Use exact text from the input for extraction. Do not paraphrase.
            """

            # Define examples for JEE math problems
            examples = [
                lx.data.ExampleData(
                    text="Problem 1: Find the derivative of f(x) = x³ + 2x² - 5x + 1. Solution: f'(x) = 3x² + 4x - 5",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="problem_number",
                            extraction_text="Problem 1",
                            attributes={"type": "identifier"}
                        ),
                        lx.data.Extraction(
                            extraction_class="question",
                            extraction_text="Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                            attributes={"topic": "calculus", "difficulty": "medium"}
                        ),
                        lx.data.Extraction(
                            extraction_class="mathematical_expression",
                            extraction_text="f(x) = x³ + 2x² - 5x + 1",
                            attributes={"type": "function"}
                        ),
                        lx.data.Extraction(
                            extraction_class="answer",
                            extraction_text="f'(x) = 3x² + 4x - 5",
                            attributes={"type": "derivative"}
                        )
                    ]
                )
            ]
            
            # Use LangExtract with proper API
            result = lx.extract(
                text_or_documents=pdf_text,
                prompt_description=prompt_description,
                examples=examples,
                model_id="gemini-2.0-flash-exp",
                api_key=self.gemini_api_key
            )
            
            processing_time = time.time() - start_time
            
            # Process the extraction results
            structured_data = {
                "extractions": [],
                "total_extractions": len(result.extractions) if hasattr(result, 'extractions') else 0,
                "text_length": len(result.text) if hasattr(result, 'text') else len(pdf_text)
            }
            
            raw_content = ""
            problems_count = 0
            math_formulas = 0
            
            if hasattr(result, 'extractions'):
                for extraction in result.extractions:
                    extraction_dict = {
                        "class": extraction.extraction_class,
                        "text": extraction.extraction_text,
                        "attributes": extraction.attributes if hasattr(extraction, 'attributes') else {}
                    }
                    structured_data["extractions"].append(extraction_dict)
                    raw_content += f"{extraction.extraction_class}: {extraction.extraction_text}\n"
                    
                    if extraction.extraction_class == "problem_number":
                        problems_count += 1
                    elif extraction.extraction_class == "mathematical_expression":
                        math_formulas += 1
            
            return ExtractionResult(
                library="LangExtract",
                processing_time=processing_time,
                success=True,
                raw_content=raw_content,
                structured_data=structured_data,
                text_length=len(raw_content),
                element_count=problems_count,
                math_formulas_detected=math_formulas,
                problems_detected=problems_count
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                library="LangExtract",
                processing_time=processing_time,
                success=False,
                raw_content="",
                structured_data={},
                error=str(e)
            )
    
    async def extract_with_llamaparse(self) -> ExtractionResult:
        """Extract using LlamaParse with markdown output"""
        print("Testing LlamaParse extraction...")
        start_time = time.time()
        
        if not LLAMAPARSE_AVAILABLE:
            return ExtractionResult(
                library="LlamaParse",
                processing_time=0,
                success=False,
                raw_content="",
                structured_data={},
                error="LlamaParse not installed"
            )
        
        if not self.llamaparse_api_key:
            return ExtractionResult(
                library="LlamaParse",
                processing_time=0,
                success=False,
                raw_content="",
                structured_data={},
                error="LLAMA_CLOUD_API_KEY not set"
            )
        
        try:
            # Configure LlamaParse for mathematical content
            parser = LlamaParse(
                api_key=self.llamaparse_api_key,
                result_type="markdown",  # Get markdown for better structure
                system_prompt="""
                Extract mathematical problems and solutions with clear structure.
                Preserve all mathematical formulas and expressions.
                Identify problem numbers, questions, solutions, and answers.
                Maintain proper formatting for mathematical notation.
                """,
                max_timeout=60000  # 60 seconds timeout
            )
            
            # Parse the document
            documents = await parser.aload_data(str(self.pdf_path))
            
            processing_time = time.time() - start_time
            
            # Process the markdown content
            full_markdown = ""
            structured_data = {
                "documents": [],
                "total_documents": len(documents),
                "parsing_instruction_used": True
            }
            
            for i, doc in enumerate(documents):
                doc_text = doc.text if hasattr(doc, 'text') else str(doc)
                full_markdown += doc_text + "\n"
                
                structured_data["documents"].append({
                    "document_id": i,
                    "text": doc_text,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                })
            
            # Analyze markdown for mathematical content
            math_indicators = ["$", "\\(", "\\[", "=", "∫", "∑", "√", "≤", "≥", "→"]
            math_formulas = sum(full_markdown.count(indicator) for indicator in math_indicators)
            
            # Problem detection in markdown
            problem_patterns = ["##", "Problem", "Q.", "Find", "Calculate", "Solve"]
            problems = sum(full_markdown.count(pattern) for pattern in problem_patterns)
            
            return ExtractionResult(
                library="LlamaParse",
                processing_time=processing_time,
                success=True,
                raw_content=full_markdown,
                structured_data=structured_data,
                text_length=len(full_markdown),
                element_count=len(documents),
                math_formulas_detected=math_formulas,
                problems_detected=problems
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                library="LlamaParse",
                processing_time=processing_time,
                success=False,
                raw_content="",
                structured_data={},
                error=str(e)
            )
    
    def extract_with_unstructured(self) -> ExtractionResult:
        """Extract using Unstructured library for AI-ready format"""
        print("Testing Unstructured extraction...")
        start_time = time.time()
        
        if not UNSTRUCTURED_AVAILABLE:
            return ExtractionResult(
                library="Unstructured",
                processing_time=0,
                success=False,
                raw_content="",
                structured_data={},
                error="Unstructured not installed"
            )
        
        try:
            # Use partition_pdf for direct PDF processing
            elements = partition_pdf(
                filename=str(self.pdf_path),
                strategy="auto",  # Let unstructured choose the best strategy
                infer_table_structure=True,  # Important for math problems
                extract_images_in_pdf=False,  # Focus on text content
                include_page_breaks=True
            )
            
            processing_time = time.time() - start_time
            
            # Process elements into structured format
            structured_data = {
                "elements": [],
                "total_elements": len(elements),
                "element_types": {},
                "pages": {}
            }
            
            full_text = ""
            
            for element in elements:
                element_dict = {
                    "type": str(type(element).__name__),
                    "text": str(element),
                    "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                }
                
                structured_data["elements"].append(element_dict)
                full_text += str(element) + "\n"
                
                # Count element types
                element_type = element_dict["type"]
                structured_data["element_types"][element_type] = \
                    structured_data["element_types"].get(element_type, 0) + 1
                
                # Group by page if available
                page_num = element_dict["metadata"].get("page_number", 1)
                if page_num not in structured_data["pages"]:
                    structured_data["pages"][page_num] = []
                structured_data["pages"][page_num].append(element_dict)
            
            # Analyze for mathematical content
            math_indicators = ["=", "∫", "∑", "√", "π", "α", "β", "γ", "θ", "∆"]
            math_formulas = sum(full_text.count(indicator) for indicator in math_indicators)
            
            # Problem detection
            problem_indicators = ["Problem", "Q.", "Find", "Calculate", "Solve", "Prove"]
            problems = sum(full_text.count(indicator) for indicator in problem_indicators)
            
            return ExtractionResult(
                library="Unstructured",
                processing_time=processing_time,
                success=True,
                raw_content=full_text,
                structured_data=structured_data,
                text_length=len(full_text),
                element_count=len(elements),
                math_formulas_detected=math_formulas,
                problems_detected=problems
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                library="Unstructured",
                processing_time=processing_time,
                success=False,
                raw_content="",
                structured_data={},
                error=str(e)
            )
    
    async def run_comparison(self) -> List[ExtractionResult]:
        """Run all extraction methods and compare results"""
        print(f"Starting PDF extraction comparison for: {self.pdf_path}")
        print("=" * 60)
        
        # Run all extraction methods
        self.results = []
        
        # PyMuPDF (synchronous)
        result_pymupdf = self.extract_with_pymupdf()
        self.results.append(result_pymupdf)
        
        # LangExtract (synchronous)
        result_langextract = self.extract_with_langextract()
        self.results.append(result_langextract)
        
        # LlamaParse (asynchronous)
        result_llamaparse = await self.extract_with_llamaparse()
        self.results.append(result_llamaparse)
        
        # Unstructured (synchronous)
        result_unstructured = self.extract_with_unstructured()
        self.results.append(result_unstructured)
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        """Generate a detailed comparison report"""
        report = []
        report.append("JEE MATH PDF EXTRACTION COMPARISON REPORT")
        report.append("=" * 50)
        report.append(f"PDF File: {self.pdf_path}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("EXTRACTION RESULTS SUMMARY:")
        report.append("-" * 30)
        
        for result in self.results:
            report.append(f"\n{result.library}:")
            report.append(f"  Status: {'✓ Success' if result.success else '✗ Failed'}")
            if result.error:
                report.append(f"  Error: {result.error}")
                continue
                
            report.append(f"  Processing Time: {result.processing_time:.2f}s")
            report.append(f"  Text Length: {result.text_length:,} characters")
            report.append(f"  Elements/Pages: {result.element_count}")
            report.append(f"  Math Formulas Detected: {result.math_formulas_detected}")
            report.append(f"  Problems Detected: {result.problems_detected}")
        
        # Performance comparison
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            report.append("\n\nPERFORMANCE RANKING:")
            report.append("-" * 20)
            
            # Speed ranking
            speed_ranking = sorted(successful_results, key=lambda x: x.processing_time)
            report.append("\nBy Speed (fastest first):")
            for i, result in enumerate(speed_ranking, 1):
                report.append(f"  {i}. {result.library}: {result.processing_time:.2f}s")
            
            # Content richness ranking
            content_ranking = sorted(successful_results, 
                                   key=lambda x: (x.problems_detected, x.math_formulas_detected, x.text_length),
                                   reverse=True)
            report.append("\nBy Content Richness (best first):")
            for i, result in enumerate(content_ranking, 1):
                report.append(f"  {i}. {result.library}: {result.problems_detected} problems, "
                            f"{result.math_formulas_detected} math elements")
        
        # Recommendations
        report.append("\n\nRECOMMENDATIONS:")
        report.append("-" * 15)
        
        if successful_results:
            best_content = max(successful_results, 
                             key=lambda x: (x.problems_detected, x.math_formulas_detected))
            fastest = min(successful_results, key=lambda x: x.processing_time)
            
            report.append(f"• Best Content Extraction: {best_content.library}")
            report.append(f"• Fastest Processing: {fastest.library}")
            
            # LangExtract specific
            langextract_result = next((r for r in successful_results if r.library == "LangExtract"), None)
            if langextract_result:
                report.append(f"• Most Structured Output: LangExtract (grounded extraction)")
            
            # Overall recommendation
            if best_content.library == fastest.library:
                report.append(f"\n• OVERALL WINNER: {best_content.library} (best content + fastest)")
            else:
                report.append(f"\n• For Quality: Choose {best_content.library}")
                report.append(f"• For Speed: Choose {fastest.library}")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "extraction_results"):
        """Save detailed results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save comparison report
        report = self.generate_comparison_report()
        report_file = output_path / f"comparison_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save individual extraction results
        for result in self.results:
            if result.success and result.raw_content:
                filename = f"{result.library.lower()}_{timestamp}.txt"
                content_file = output_path / filename
                with open(content_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== {result.library} Extraction Result ===\n\n")
                    f.write(result.raw_content)
        
        # Save structured data as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                "library": result.library,
                "success": result.success,
                "processing_time": result.processing_time,
                "text_length": result.text_length,
                "element_count": result.element_count,
                "math_formulas_detected": result.math_formulas_detected,
                "problems_detected": result.problems_detected,
                "error": result.error,
                "structured_data": result.structured_data if result.success else None
            })
        
        json_file = output_path / f"results_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        print(f"- Report: {report_file}")
        print(f"- Data: {json_file}")

async def main():
    """Main function to run the comparison"""
    # Default PDF path - update this to your JEE Math PDF
    default_pdf = "path/to/your/jee_math.pdf"
    
    # Check if PDF file exists
    if len(os.sys.argv) > 1:
        pdf_path = os.sys.argv[1]
    else:
        pdf_path = default_pdf
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Usage: python jee_extraction_comparison.py <path_to_pdf>")
        return
    
    # Create comparison instance
    comparison = JEEExtractionComparison(pdf_path)
    
    # Run the comparison
    results = await comparison.run_comparison()
    
    # Print the report
    print("\n" + comparison.generate_comparison_report())
    
    # Save results
    comparison.save_results()
    
    # Return the best method
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_result = max(successful_results, 
                         key=lambda x: (x.problems_detected, x.math_formulas_detected))
        print(f"\n🏆 RECOMMENDED LIBRARY: {best_result.library}")
        print(f"   Reason: Best content extraction with {best_result.problems_detected} problems detected")
    else:
        print("\n❌ No extraction methods succeeded. Check your API keys and dependencies.")

if __name__ == "__main__":
    # Check for required environment variables
    print("Checking environment setup...")
    
    api_keys_status = []
    if os.getenv('GOOGLE_API_KEY'):
        api_keys_status.append("✓ GOOGLE_API_KEY set (for LangExtract)")
    else:
        api_keys_status.append("✗ GOOGLE_API_KEY missing (LangExtract will be skipped)")
    
    if os.getenv('LLAMA_CLOUD_API_KEY'):
        api_keys_status.append("✓ LLAMA_CLOUD_API_KEY set (for LlamaParse)")
    else:
        api_keys_status.append("✗ LLAMA_CLOUD_API_KEY missing (LlamaParse will be skipped)")
    
    print("\n".join(api_keys_status))
    print()
    
    # Run the async main function
    asyncio.run(main())
