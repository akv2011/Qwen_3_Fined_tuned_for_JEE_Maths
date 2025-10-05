#!/usr/bin/env python3
"""
JEE Mathematics PDF Extraction Script
====================================

Based on analysis results, this script uses PDFPlumber (best performing library)
to extract structured data from "41 Years IIT JEE Mathematics by Amit M Agarwal.pdf"

Key findings from analysis:
- 625 total pages
- 9 tables detected in first 3 pages
- PDFPlumber best for structure preservation
- Need custom math notation handling
"""

import os
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pdfplumber
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class JEEProblem:
    """Structure for a JEE math problem."""
    id: str
    chapter: str
    topic: str
    year: int
    exam_type: str  # "JEE Main", "JEE Advanced", "IIT-JEE"
    question_text: str
    options: List[str]
    correct_answer: str
    solution: str
    difficulty: str
    page_number: int
    problem_number: int
    math_notation: List[str]  # Detected math symbols/expressions

class JEEMathExtractor:
    """Extract structured JEE problems from PDF using PDFPlumber."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.problems = []
        self.current_chapter = ""
        self.current_topic = ""
        
        # Math notation patterns
        self.math_patterns = {
            'fractions': r'\\frac\{[^}]+\}\{[^}]+\}',
            'square_root': r'√[^\\s]+|\\sqrt\{[^}]+\}',
            'integrals': r'∫[^\\s]+|\\int[^\\s]+',
            'summation': r'∑[^\\s]+|\\sum[^\\s]+',
            'greek_letters': r'[αβγδεζηθικλμνξοπρστυφχψω]',
            'superscript': r'[²³⁴⁵⁶⁷⁸⁹⁰]|\\^{[^}]+}',
            'subscript': r'[₀₁₂₃₄₅₆₇₈₉]|\\_\{[^}]+\}',
            'equations': r'[=≠≤≥<>±∓×÷]',
            'functions': r'sin|cos|tan|log|ln|exp|lim'
        }
        
        # Problem detection patterns
        self.problem_patterns = {
            'question_start': r'^(\d+)\.\s+',  # "1. ", "2. ", etc.
            'options': r'^\([A-D]\)|^\([a-d]\)|^[A-D]\)|^[a-d]\)',
            'solution': r'Sol\.?\s*[:\-]?|Solution\s*[:\-]?|Answer\s*[:\-]?',
            'year_exam': r'(JEE|IIT)[\s\-]*(Main|Advanced|\d{4})',
            'chapter_heading': r'^CHAPTER\s+\d+|^Chapter\s+\d+',
            'topic_heading': r'^[A-Z\s]+$'  # All caps topic headings
        }
    
    def detect_math_notation(self, text: str) -> List[str]:
        """Detect mathematical notation in text."""
        detected = []
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected.extend([f"{pattern_name}: {match}" for match in matches])
        return detected
    
    def extract_page_content(self, page) -> Dict[str, Any]:
        """Extract structured content from a single page."""
        try:
            # Extract text
            text = page.extract_text() or ""
            
            # Extract tables
            tables = page.extract_tables()
            
            # Get character-level details for fine structure
            chars = page.chars
            
            # Detect layout elements
            lines = page.lines
            rects = page.rects
            
            return {
                'page_number': page.page_number,
                'text': text,
                'text_length': len(text),
                'tables': tables,
                'table_count': len(tables),
                'chars_count': len(chars),
                'lines_count': len(lines),
                'rects_count': len(rects),
                'math_notation': self.detect_math_notation(text),
                'problems_detected': self.detect_problems_on_page(text),
                'chapter_topic_info': self.extract_chapter_topic(text)
            }
        except Exception as e:
            return {
                'page_number': page.page_number,
                'error': str(e),
                'text': '',
                'tables': [],
                'problems_detected': []
            }
    
    def extract_chapter_topic(self, text: str) -> Dict[str, str]:
        """Extract chapter and topic information from page text."""
        info = {'chapter': '', 'topic': ''}
        
        # Look for chapter headings
        chapter_match = re.search(self.problem_patterns['chapter_heading'], text, re.MULTILINE)
        if chapter_match:
            # Extract chapter name from next line or same line
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if re.search(self.problem_patterns['chapter_heading'], line):
                    if i + 1 < len(lines):
                        info['chapter'] = lines[i + 1].strip()
                    break
        
        # Look for topic headings (usually all caps)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(self.problem_patterns['topic_heading'], line) and len(line) > 5:
                if not any(char.islower() for char in line):  # All uppercase
                    info['topic'] = line
                    break
        
        return info
    
    def detect_problems_on_page(self, text: str) -> List[Dict[str, Any]]:
        """Detect individual problems on a page."""
        problems = []
        lines = text.split('\n')
        
        current_problem = None
        current_section = 'question'  # question, options, solution
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for problem start
            problem_match = re.match(self.problem_patterns['question_start'], line)
            if problem_match:
                # Save previous problem if exists
                if current_problem:
                    problems.append(current_problem)
                
                # Start new problem
                problem_num = int(problem_match.group(1))
                current_problem = {
                    'problem_number': problem_num,
                    'question_text': line,
                    'options': [],
                    'solution': '',
                    'year_exam': '',
                    'line_start': i
                }
                current_section = 'question'
                continue
            
            if current_problem:
                # Check for options
                if re.match(self.problem_patterns['options'], line):
                    current_problem['options'].append(line)
                    current_section = 'options'
                    continue
                
                # Check for solution
                if re.search(self.problem_patterns['solution'], line, re.IGNORECASE):
                    current_problem['solution'] = line
                    current_section = 'solution'
                    continue
                
                # Check for year/exam info
                year_exam_match = re.search(self.problem_patterns['year_exam'], line)
                if year_exam_match:
                    current_problem['year_exam'] = year_exam_match.group(0)
                    continue
                
                # Add to current section
                if current_section == 'question':
                    current_problem['question_text'] += ' ' + line
                elif current_section == 'solution':
                    current_problem['solution'] += ' ' + line
        
        # Add last problem
        if current_problem:
            problems.append(current_problem)
        
        return problems
    
    def extract_sample_pages(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Extract and analyze sample pages to understand structure."""
        sample_data = []
        
        print(f"📖 Extracting sample pages (first {max_pages} pages)...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_process = min(max_pages, total_pages)
            
            for i in range(pages_to_process):
                page = pdf.pages[i]
                print(f"   Processing page {i + 1}/{pages_to_process}...")
                
                page_data = self.extract_page_content(page)
                sample_data.append(page_data)
                
                # Update chapter/topic info
                chapter_topic = page_data.get('chapter_topic_info', {})
                if chapter_topic.get('chapter'):
                    self.current_chapter = chapter_topic['chapter']
                if chapter_topic.get('topic'):
                    self.current_topic = chapter_topic['topic']
        
        return sample_data
    
    def analyze_extraction_quality(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality of extraction from sample pages."""
        total_pages = len(sample_data)
        total_problems = sum(len(page.get('problems_detected', [])) for page in sample_data)
        total_tables = sum(page.get('table_count', 0) for page in sample_data)
        total_math_notation = sum(len(page.get('math_notation', [])) for page in sample_data)
        
        pages_with_problems = sum(1 for page in sample_data if page.get('problems_detected', []))
        pages_with_tables = sum(1 for page in sample_data if page.get('table_count', 0) > 0)
        pages_with_math = sum(1 for page in sample_data if page.get('math_notation', []))
        
        analysis = {
            'total_pages_analyzed': total_pages,
            'total_problems_detected': total_problems,
            'total_tables_detected': total_tables,
            'total_math_notation': total_math_notation,
            'pages_with_problems': pages_with_problems,
            'pages_with_tables': pages_with_tables,
            'pages_with_math': pages_with_math,
            'average_problems_per_page': total_problems / total_pages if total_pages > 0 else 0,
            'average_tables_per_page': total_tables / total_pages if total_pages > 0 else 0,
            'problem_detection_rate': pages_with_problems / total_pages if total_pages > 0 else 0,
            'table_detection_rate': pages_with_tables / total_pages if total_pages > 0 else 0,
            'math_detection_rate': pages_with_math / total_pages if total_pages > 0 else 0
        }
        
        return analysis
    
    def save_sample_analysis(self, sample_data: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save sample analysis results."""
        
        # Save detailed sample data
        with open('jee_sample_extraction.json', 'w', encoding='utf-8') as f:
            json.dump({
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'pdf_path': self.pdf_path,
                'analysis_summary': analysis,
                'sample_pages': sample_data
            }, f, indent=2, ensure_ascii=False)
        
        # Create readable summary
        with open('jee_sample_summary.txt', 'w', encoding='utf-8') as f:
            f.write("JEE Mathematics PDF Sample Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"PDF: {os.path.basename(self.pdf_path)}\n")
            f.write(f"Total pages analyzed: {analysis['total_pages_analyzed']}\n\n")
            
            f.write("Detection Statistics:\n")
            f.write(f"- Problems detected: {analysis['total_problems_detected']}\n")
            f.write(f"- Tables detected: {analysis['total_tables_detected']}\n")
            f.write(f"- Math notation instances: {analysis['total_math_notation']}\n\n")
            
            f.write("Detection Rates:\n")
            f.write(f"- Pages with problems: {analysis['problem_detection_rate']:.1%}\n")
            f.write(f"- Pages with tables: {analysis['table_detection_rate']:.1%}\n")
            f.write(f"- Pages with math: {analysis['math_detection_rate']:.1%}\n\n")
            
            f.write("Sample Problem Structure:\n")
            for i, page in enumerate(sample_data[:3]):  # First 3 pages
                f.write(f"\nPage {page['page_number']}:\n")
                f.write(f"  Text length: {page['text_length']} chars\n")
                f.write(f"  Problems: {len(page.get('problems_detected', []))}\n")
                f.write(f"  Tables: {page.get('table_count', 0)}\n")
                f.write(f"  Math notation: {len(page.get('math_notation', []))}\n")
                
                # Show first problem if exists
                problems = page.get('problems_detected', [])
                if problems:
                    prob = problems[0]
                    f.write(f"  First problem: {prob.get('question_text', '')[:100]}...\n")
        
        print(f"✅ Sample analysis saved to:")
        print(f"   - jee_sample_extraction.json (detailed)")
        print(f"   - jee_sample_summary.txt (readable)")
    
    def create_extraction_demo(self) -> Dict[str, Any]:
        """Create a demonstration of the extraction capabilities."""
        
        print("\n🔍 JEE Mathematics PDF Extraction Demo")
        print("=" * 50)
        
        # Extract sample pages
        sample_data = self.extract_sample_pages(max_pages=10)
        
        # Analyze quality
        analysis = self.analyze_extraction_quality(sample_data)
        
        # Save results
        self.save_sample_analysis(sample_data, analysis)
        
        # Create demo summary
        demo_results = {
            'extraction_method': 'PDFPlumber (recommended from analysis)',
            'sample_pages_processed': len(sample_data),
            'key_findings': [],
            'extraction_capabilities': [],
            'recommendations': []
        }
        
        # Add key findings
        if analysis['total_problems_detected'] > 0:
            demo_results['key_findings'].append(f"Detected {analysis['total_problems_detected']} problems across {analysis['total_pages_analyzed']} pages")
        
        if analysis['total_tables_detected'] > 0:
            demo_results['key_findings'].append(f"Found {analysis['total_tables_detected']} tables (likely solution steps)")
        
        if analysis['total_math_notation'] > 0:
            demo_results['key_findings'].append(f"Identified {analysis['total_math_notation']} mathematical notation instances")
        
        # Add capabilities
        demo_results['extraction_capabilities'] = [
            "Problem number detection",
            "Question text extraction",
            "Multiple choice options parsing",
            "Solution text identification",
            "Table extraction for step-by-step solutions",
            "Mathematical notation recognition",
            "Chapter/topic structure analysis"
        ]
        
        # Add recommendations
        demo_results['recommendations'] = [
            "PDFPlumber is optimal for this PDF structure",
            "Implement problem boundary detection",
            "Develop math notation cleaning pipeline",
            "Use table extraction for solution steps",
            "Create validation rules for extracted problems",
            "Consider OCR for complex mathematical formulas"
        ]
        
        return demo_results

def main():
    """Main extraction demo function."""
    
    # Find PDF file
    pdf_path = None
    current_dir = Path.cwd()
    
    for pdf_file in current_dir.glob("*.pdf"):
        if "JEE" in pdf_file.name or "Mathematics" in pdf_file.name:
            pdf_path = str(pdf_file)
            break
    
    if not pdf_path:
        print("❌ Could not find JEE Mathematics PDF file")
        return
    
    print(f"✅ Found PDF: {os.path.basename(pdf_path)}")
    
    # Create extractor
    extractor = JEEMathExtractor(pdf_path)
    
    # Run extraction demo
    demo_results = extractor.create_extraction_demo()
    
    # Print summary
    print("\n🎯 Extraction Demo Results")
    print("=" * 30)
    
    print(f"📊 Method: {demo_results['extraction_method']}")
    print(f"📄 Pages processed: {demo_results['sample_pages_processed']}")
    
    print("\n🔍 Key Findings:")
    for finding in demo_results['key_findings']:
        print(f"   • {finding}")
    
    print("\n⚡ Extraction Capabilities:")
    for capability in demo_results['extraction_capabilities']:
        print(f"   • {capability}")
    
    print("\n💡 Recommendations:")
    for rec in demo_results['recommendations']:
        print(f"   • {rec}")
    
    print("\n🚀 Next Steps:")
    print("   1. Review sample extraction results")
    print("   2. Refine problem detection patterns")
    print("   3. Implement full PDF processing pipeline")
    print("   4. Create structured dataset for model training")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
