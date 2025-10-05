#!/usr/bin/env python3
"""
Deep Analysis of JEE Mathematics PDF Structure
=============================================

This script performs comprehensive analysis of "41 Years IIT JEE Mathematics by Amit M Agarwal.pdf"
using multiple PDF processing libraries to determine the best extraction approach for creating 
a structured JEE math dataset.

Libraries tested:
1. PyPDF2/pypdf - Basic text extraction
2. Unstructured - Advanced layout analysis with ML
3. LangChain PDF loaders - Multiple strategies
4. PyMuPDF (fitz) - Layout-aware extraction
5. PDFPlumber - Table and structure detection

This analysis will help determine:
- Best library for extracting math problems
- How to preserve problem structure
- Methods for handling mathematical notation
- Approach for question/answer separation
- Table extraction for solution steps
"""

import os
import sys
import traceback
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional

# Check and install required packages
def check_and_install_packages():
    """Check for required packages and install if missing."""
    required_packages = [
        'pypdf',
        'unstructured', 
        'langchain-community',
        'PyMuPDF',
        'pdfplumber',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PyMuPDF':
                import fitz
            elif package == 'unstructured':
                from unstructured.partition.pdf import partition_pdf
            elif package == 'langchain-community':
                from langchain_community.document_loaders import PyPDFLoader
            elif package == 'pypdf':
                import pypdf
            elif package == 'pdfplumber':
                import pdfplumber
            elif package == 'Pillow':
                from PIL import Image
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
        print("🎉 All packages installed successfully!")
    
    return True

def find_pdf_file() -> Optional[str]:
    """Find the JEE Mathematics PDF file."""
    current_dir = Path.cwd()
    possible_names = [
        "41 Years IIT JEE Mathematics by Amit M Agarwal.pdf",
        "JEE_Mathematics.pdf",
        "*.pdf"
    ]
    
    # Search in current directory
    for pattern in possible_names:
        if "*" in pattern:
            pdf_files = list(current_dir.glob(pattern))
            if pdf_files:
                return str(pdf_files[0])  # Return first PDF found
        else:
            pdf_path = current_dir / pattern
            if pdf_path.exists():
                return str(pdf_path)
    
    return None

def analyze_with_pypdf(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF using pypdf (basic text extraction)."""
    print("\n📖 Analyzing with PyPDF...")
    
    try:
        import pypdf
        
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            results = {
                'library': 'pypdf',
                'total_pages': len(reader.pages),
                'metadata': dict(reader.metadata) if reader.metadata else {},
                'sample_pages': [],
                'pros': [],
                'cons': [],
                'math_detection': False
            }
            
            # Analyze first 3 pages
            for i in range(min(3, len(reader.pages))):
                page = reader.pages[i]
                text = page.extract_text()
                
                page_info = {
                    'page_number': i + 1,
                    'text_length': len(text),
                    'text_preview': text[:500] + "..." if len(text) > 500 else text,
                    'has_math_symbols': any(symbol in text for symbol in ['∫', '∑', '√', '²', '³', 'α', 'β', 'π', '∆', '∂'])
                }
                results['sample_pages'].append(page_info)
                
                if page_info['has_math_symbols']:
                    results['math_detection'] = True
            
            # Assess pros and cons
            results['pros'] = [
                'Fast and lightweight',
                'Good basic text extraction',
                'Preserves some structure'
            ]
            results['cons'] = [
                'Limited layout analysis',
                'May lose mathematical formatting',
                'No table detection',
                'No image extraction'
            ]
            
            return results
            
    except Exception as e:
        return {
            'library': 'pypdf',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def analyze_with_unstructured(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF using Unstructured (advanced ML-based extraction)."""
    print("\n🧠 Analyzing with Unstructured...")
    
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.partition.auto import partition
        
        # Use partition_pdf for detailed analysis
        elements = partition_pdf(
            filename=pdf_path,
            pages=[1, 2, 3],  # First 3 pages for analysis
            infer_table_structure=True,
            strategy="hi_res"  # High resolution for better math extraction
        )
        
        results = {
            'library': 'unstructured',
            'total_elements': len(elements),
            'element_types': {},
            'sample_elements': [],
            'tables_detected': 0,
            'math_detection': False,
            'pros': [],
            'cons': []
        }
        
        # Analyze elements
        for element in elements[:10]:  # First 10 elements
            element_type = type(element).__name__
            results['element_types'][element_type] = results['element_types'].get(element_type, 0) + 1
            
            element_info = {
                'type': element_type,
                'text_preview': str(element)[:200] + "..." if len(str(element)) > 200 else str(element),
                'metadata': getattr(element, 'metadata', {}).to_dict() if hasattr(getattr(element, 'metadata', {}), 'to_dict') else {}
            }
            results['sample_elements'].append(element_info)
            
            # Check for math symbols
            if any(symbol in str(element) for symbol in ['∫', '∑', '√', '²', '³', 'α', 'β', 'π', '∆', '∂', '=']):
                results['math_detection'] = True
            
            # Count tables
            if 'Table' in element_type:
                results['tables_detected'] += 1
        
        results['pros'] = [
            'Advanced ML-based layout detection',
            'Excellent table extraction',
            'Preserves document structure',
            'Handles complex layouts',
            'Good for academic papers'
        ]
        results['cons'] = [
            'Slower processing',
            'Requires more dependencies',
            'May need fine-tuning for math notation'
        ]
        
        return results
        
    except Exception as e:
        return {
            'library': 'unstructured',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def analyze_with_langchain(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF using LangChain loaders."""
    print("\n🔗 Analyzing with LangChain...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
        
        results = {
            'library': 'langchain',
            'loaders_tested': [],
            'pros': [],
            'cons': []
        }
        
        # Test PyPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            pypdf_result = {
                'loader_name': 'PyPDFLoader',
                'total_documents': len(docs),
                'sample_content': docs[0].page_content[:300] + "..." if docs else "No content",
                'metadata_sample': docs[0].metadata if docs else {},
                'math_detection': any(symbol in docs[0].page_content for symbol in ['∫', '∑', '√', '²', '³'] if docs)
            }
            results['loaders_tested'].append(pypdf_result)
        except Exception as e:
            results['loaders_tested'].append({
                'loader_name': 'PyPDFLoader',
                'error': str(e)
            })
        
        # Test UnstructuredPDFLoader
        try:
            loader = UnstructuredPDFLoader(pdf_path)
            docs = loader.load()
            
            unstructured_result = {
                'loader_name': 'UnstructuredPDFLoader',
                'total_documents': len(docs),
                'sample_content': docs[0].page_content[:300] + "..." if docs else "No content",
                'metadata_sample': docs[0].metadata if docs else {},
                'math_detection': any(symbol in docs[0].page_content for symbol in ['∫', '∑', '√', '²', '³'] if docs)
            }
            results['loaders_tested'].append(unstructured_result)
        except Exception as e:
            results['loaders_tested'].append({
                'loader_name': 'UnstructuredPDFLoader',
                'error': str(e)
            })
        
        results['pros'] = [
            'Easy integration with LLM workflows',
            'Multiple loader options',
            'Consistent document format',
            'Good for RAG pipelines'
        ]
        results['cons'] = [
            'Limited customization',
            'Dependent on underlying libraries',
            'May not preserve fine structure'
        ]
        
        return results
        
    except Exception as e:
        return {
            'library': 'langchain',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def analyze_with_pymupdf(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF using PyMuPDF (fitz) for layout-aware extraction."""
    print("\n📋 Analyzing with PyMuPDF...")
    
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        
        results = {
            'library': 'pymupdf',
            'total_pages': len(doc),
            'metadata': doc.metadata,
            'sample_pages': [],
            'images_detected': 0,
            'tables_detected': 0,
            'math_detection': False,
            'pros': [],
            'cons': []
        }
        
        # Analyze first 3 pages
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            
            # Extract text with layout info
            text = page.get_text()
            text_dict = page.get_text("dict")
            
            # Count images
            images = page.get_images()
            results['images_detected'] += len(images)
            
            # Try to detect tables (basic approach)
            tables = page.find_tables()
            results['tables_detected'] += len(tables)
            
            page_info = {
                'page_number': page_num + 1,
                'text_length': len(text),
                'text_preview': text[:500] + "..." if len(text) > 500 else text,
                'images_count': len(images),
                'tables_count': len(tables),
                'blocks_count': len(text_dict.get('blocks', [])),
                'has_math_symbols': any(symbol in text for symbol in ['∫', '∑', '√', '²', '³', 'α', 'β', 'π', '∆', '∂'])
            }
            results['sample_pages'].append(page_info)
            
            if page_info['has_math_symbols']:
                results['math_detection'] = True
        
        doc.close()
        
        results['pros'] = [
            'Excellent layout preservation',
            'Good table detection',
            'Image extraction capabilities',
            'Font and formatting info',
            'Fast processing'
        ]
        results['cons'] = [
            'Complex API for advanced use',
            'May need post-processing for structure',
            'Math notation may need special handling'
        ]
        
        return results
        
    except Exception as e:
        return {
            'library': 'pymupdf',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def analyze_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF using pdfplumber for table and structure detection."""
    print("\n🔍 Analyzing with PDFPlumber...")
    
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            results = {
                'library': 'pdfplumber',
                'total_pages': len(pdf.pages),
                'metadata': pdf.metadata,
                'sample_pages': [],
                'tables_detected': 0,
                'math_detection': False,
                'pros': [],
                'cons': []
            }
            
            # Analyze first 3 pages
            for i in range(min(3, len(pdf.pages))):
                page = pdf.pages[i]
                
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables()
                results['tables_detected'] += len(tables)
                
                # Get page objects info
                chars = page.chars
                lines = page.lines
                rects = page.rects
                
                page_info = {
                    'page_number': i + 1,
                    'text_length': len(text),
                    'text_preview': text[:500] + "..." if len(text) > 500 else text,
                    'tables_count': len(tables),
                    'chars_count': len(chars),
                    'lines_count': len(lines),
                    'rects_count': len(rects),
                    'has_math_symbols': any(symbol in text for symbol in ['∫', '∑', '√', '²', '³', 'α', 'β', 'π', '∆', '∂'])
                }
                results['sample_pages'].append(page_info)
                
                if page_info['has_math_symbols']:
                    results['math_detection'] = True
        
        results['pros'] = [
            'Excellent table extraction',
            'Detailed character-level info',
            'Good for structured documents',
            'Precise coordinate extraction',
            'Visual debugging capabilities'
        ]
        results['cons'] = [
            'May struggle with complex layouts',
            'Slower for large documents',
            'Limited ML-based understanding'
        ]
        
        return results
        
    except Exception as e:
        return {
            'library': 'pdfplumber',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_recommendation(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate recommendations based on analysis results."""
    
    print("\n🎯 Generating Recommendations...")
    
    # Score each library
    library_scores = {}
    
    for result in analysis_results:
        if 'error' in result:
            continue
            
        library = result['library']
        score = 0
        strengths = []
        weaknesses = []
        
        # Math detection bonus
        if result.get('math_detection', False):
            score += 20
            strengths.append("Detected mathematical notation")
        else:
            weaknesses.append("Limited math symbol detection")
        
        # Table detection bonus
        tables_detected = result.get('tables_detected', 0)
        if tables_detected > 0:
            score += 15
            strengths.append(f"Detected {tables_detected} tables")
        
        # Structure preservation
        if library == 'unstructured':
            score += 25  # Best for academic documents
            strengths.append("ML-based layout analysis")
        elif library == 'pymupdf':
            score += 20  # Good layout preservation
            strengths.append("Excellent layout preservation")
        elif library == 'pdfplumber':
            score += 18  # Good for tables
            strengths.append("Precise table extraction")
        
        # Speed consideration
        if library == 'pypdf':
            score += 10  # Fastest
            strengths.append("Fast processing")
        elif library == 'pymupdf':
            score += 8
            strengths.append("Good performance")
        
        library_scores[library] = {
            'score': score,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    # Sort by score
    sorted_libraries = sorted(library_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Generate specific recommendations for JEE Math extraction
    recommendations = {
        'best_overall': sorted_libraries[0][0] if sorted_libraries else None,
        'library_rankings': sorted_libraries,
        'extraction_strategy': {
            'primary_method': None,
            'preprocessing_steps': [],
            'postprocessing_steps': [],
            'structure_preservation': [],
            'math_handling': []
        },
        'implementation_plan': []
    }
    
    if sorted_libraries:
        best_lib = sorted_libraries[0][0]
        
        if best_lib == 'unstructured':
            recommendations['extraction_strategy'] = {
                'primary_method': 'Unstructured with hi_res strategy',
                'preprocessing_steps': [
                    'Clean PDF if needed',
                    'Ensure high DPI for math symbols'
                ],
                'postprocessing_steps': [
                    'Parse elements by type',
                    'Reconstruct problem structure',
                    'Extract tables as solution steps'
                ],
                'structure_preservation': [
                    'Use element metadata for positioning',
                    'Group related elements',
                    'Preserve problem numbering'
                ],
                'math_handling': [
                    'Extract math as-is from text',
                    'Use OCR for complex formulas if needed',
                    'Clean up common math notation issues'
                ]
            }
        elif best_lib == 'pymupdf':
            recommendations['extraction_strategy'] = {
                'primary_method': 'PyMuPDF with layout analysis',
                'preprocessing_steps': [
                    'Analyze document structure',
                    'Identify problem boundaries'
                ],
                'postprocessing_steps': [
                    'Extract text with position info',
                    'Group by problem sections',
                    'Parse tables for solutions'
                ],
                'structure_preservation': [
                    'Use coordinate information',
                    'Detect section headings',
                    'Maintain problem hierarchy'
                ],
                'math_handling': [
                    'Extract with font information',
                    'Identify math symbols by font',
                    'Preserve mathematical layout'
                ]
            }
    
    # Implementation steps
    recommendations['implementation_plan'] = [
        f"1. Use {recommendations['best_overall']} as primary extraction method",
        "2. Extract first 10 pages to validate approach",
        "3. Develop problem detection regex patterns",
        "4. Create structured output schema",
        "5. Implement batch processing pipeline",
        "6. Add quality validation checks",
        "7. Test with different JEE problem types"
    ]
    
    return recommendations

def save_analysis_report(analysis_results: List[Dict[str, Any]], recommendations: Dict[str, Any], pdf_path: str):
    """Save complete analysis report."""
    
    report = {
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pdf_analyzed': pdf_path,
        'pdf_size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
        'library_analyses': analysis_results,
        'recommendations': recommendations,
        'next_steps': [
            "Review the recommended extraction strategy",
            "Test the primary method on a few pages",
            "Develop problem structure schema",
            "Create validation pipeline",
            "Implement full extraction script"
        ]
    }
    
    # Save to JSON file
    report_path = "pdf_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed analysis report saved to: {report_path}")
    
    # Create summary markdown
    summary_path = "pdf_analysis_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# JEE Mathematics PDF Analysis Report\n\n")
        f.write(f"**Analysis Date:** {report['analysis_timestamp']}\n")
        f.write(f"**PDF File:** {os.path.basename(pdf_path)}\n")
        f.write(f"**File Size:** {report['pdf_size_mb']} MB\n\n")
        
        f.write("## 🏆 Recommendations\n\n")
        f.write(f"**Best Library:** {recommendations['best_overall']}\n\n")
        
        f.write("### Library Rankings:\n")
        for i, (lib, data) in enumerate(recommendations['library_rankings'], 1):
            f.write(f"{i}. **{lib}** (Score: {data['score']})\n")
            f.write(f"   - Strengths: {', '.join(data['strengths'])}\n")
            if data['weaknesses']:
                f.write(f"   - Weaknesses: {', '.join(data['weaknesses'])}\n")
            f.write("\n")
        
        f.write("### Extraction Strategy:\n")
        strategy = recommendations['extraction_strategy']
        f.write(f"- **Primary Method:** {strategy['primary_method']}\n")
        f.write(f"- **Preprocessing:** {', '.join(strategy['preprocessing_steps'])}\n")
        f.write(f"- **Postprocessing:** {', '.join(strategy['postprocessing_steps'])}\n")
        f.write(f"- **Structure:** {', '.join(strategy['structure_preservation'])}\n")
        f.write(f"- **Math Handling:** {', '.join(strategy['math_handling'])}\n\n")
        
        f.write("## 📊 Library Analysis Summary\n\n")
        for result in analysis_results:
            if 'error' in result:
                f.write(f"### ❌ {result['library']} (Error)\n")
                f.write(f"Error: {result['error']}\n\n")
            else:
                f.write(f"### ✅ {result['library']}\n")
                if 'total_pages' in result:
                    f.write(f"- Pages analyzed: {min(3, result['total_pages'])}\n")
                if 'math_detection' in result:
                    f.write(f"- Math detection: {'Yes' if result['math_detection'] else 'No'}\n")
                if 'tables_detected' in result:
                    f.write(f"- Tables detected: {result['tables_detected']}\n")
                f.write(f"- Pros: {', '.join(result.get('pros', []))}\n")
                f.write(f"- Cons: {', '.join(result.get('cons', []))}\n\n")
    
    print(f"📄 Analysis summary saved to: {summary_path}")

def main():
    """Main analysis function."""
    print("🎯 JEE Mathematics PDF Deep Analysis")
    print("=" * 50)
    
    # Check and install packages
    if not check_and_install_packages():
        print("❌ Failed to install required packages. Exiting.")
        return
    
    # Find PDF file
    pdf_path = find_pdf_file()
    if not pdf_path:
        print("❌ Could not find JEE Mathematics PDF file.")
        print("Please ensure the PDF is in the current directory.")
        return
    
    print(f"✅ Found PDF: {os.path.basename(pdf_path)}")
    print(f"📊 File size: {round(os.path.getsize(pdf_path) / (1024 * 1024), 2)} MB")
    
    # Run analyses
    analysis_results = []
    
    # Test each library
    libraries = [
        ('PyPDF', analyze_with_pypdf),
        ('Unstructured', analyze_with_unstructured),
        ('LangChain', analyze_with_langchain),
        ('PyMuPDF', analyze_with_pymupdf),
        ('PDFPlumber', analyze_with_pdfplumber)
    ]
    
    for name, analyzer_func in libraries:
        print(f"\n🔍 Testing {name}...")
        try:
            result = analyzer_func(pdf_path)
            analysis_results.append(result)
            
            if 'error' in result:
                print(f"❌ {name} failed: {result['error']}")
            else:
                print(f"✅ {name} completed successfully")
                if 'math_detection' in result:
                    print(f"   Math symbols detected: {'Yes' if result['math_detection'] else 'No'}")
                if 'tables_detected' in result:
                    print(f"   Tables found: {result['tables_detected']}")
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            analysis_results.append({
                'library': name.lower(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    # Generate recommendations
    recommendations = generate_recommendation(analysis_results)
    
    # Save results
    save_analysis_report(analysis_results, recommendations, pdf_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 50)
    
    if recommendations['best_overall']:
        print(f"🏆 Recommended Library: {recommendations['best_overall']}")
        print(f"📋 Strategy: {recommendations['extraction_strategy']['primary_method']}")
        
        print("\n📝 Next Steps:")
        for step in recommendations['implementation_plan'][:3]:
            print(f"   {step}")
        
        print(f"\n📄 Full report saved to:")
        print(f"   - pdf_analysis_report.json")
        print(f"   - pdf_analysis_summary.md")
    else:
        print("❌ No libraries were successfully analyzed.")
    
    print("\n🚀 Ready to implement the recommended approach!")

if __name__ == "__main__":
    main()
