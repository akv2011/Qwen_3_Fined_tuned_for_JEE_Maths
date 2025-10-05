#!/usr/bin/env python3
"""
Local JEE Math Dataset Creator
Uses LangChain to parse PDFs locally and create structured dataset
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports for PDF parsing
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredPDFLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import pandas as pd
from pydantic import BaseModel, Field

class JEEProblem(BaseModel):
    """Structured format for a JEE math problem"""
    id: str = Field(description="Unique problem identifier")
    subject: str = Field(description="Math subject (algebra, calculus, etc.)")
    topic: str = Field(description="Specific topic within subject")
    difficulty: str = Field(description="easy, medium, hard")
    problem_text: str = Field(description="The problem statement")
    solution: str = Field(description="Step-by-step solution")
    answer: str = Field(description="Final numerical/algebraic answer")
    source_file: str = Field(description="Source PDF filename")
    page_number: int = Field(description="Page number in source")
    created_at: str = Field(description="Timestamp of extraction")

class LocalDatasetCreator:
    """Creates JEE math dataset from local PDFs using LangChain"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_database()
        
    def setup_directories(self):
        """Create local data directory structure"""
        directories = [
            "raw_pdfs",
            "processed", 
            "structured",
            "statistics"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def setup_database(self):
        """Setup local SQLite database for fast querying"""
        self.db_path = self.base_dir / "jee_problems.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS problems (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    topic TEXT,
                    difficulty TEXT,
                    problem_text TEXT,
                    solution TEXT,
                    answer TEXT,
                    source_file TEXT,
                    page_number INTEGER,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subject ON problems(subject)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_difficulty ON problems(difficulty)
            """)
        print(f"✅ Database setup complete: {self.db_path}")
    
    def parse_pdf_with_langchain(self, pdf_path: Path, loader_type: str = "pymupdf") -> List[Document]:
        """Parse PDF using specified LangChain loader"""
        print(f"📄 Parsing PDF: {pdf_path.name} with {loader_type}")
        
        try:
            if loader_type == "pymupdf":
                # Best for math PDFs with tables and complex layouts
                loader = PyMuPDFLoader(
                    str(pdf_path),
                    mode="page",
                    extract_tables="markdown"  # Extract tables as markdown
                )
            elif loader_type == "unstructured":
                # Good for mixed content and advanced parsing
                loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
            else:
                # Fallback to basic PyPDF
                loader = PyPDFLoader(str(pdf_path), mode="page")
            
            documents = loader.load()
            print(f"✅ Extracted {len(documents)} pages from {pdf_path.name}")
            
            return documents
            
        except Exception as e:
            print(f"❌ Failed to parse {pdf_path.name}: {e}")
            return []
    
    def extract_problems_from_text(self, text: str, source_file: str, page_num: int) -> List[JEEProblem]:
        """Extract structured problems from raw text (placeholder for AI extraction)"""
        # This is a simplified version - in practice, you'd use:
        # 1. LLM-based extraction (OpenAI, Anthropic, local models)
        # 2. Regex patterns for specific formats
        # 3. React agents for intelligent parsing
        
        problems = []
        
        # Simple pattern matching (replace with sophisticated extraction)
        if "problem" in text.lower() and "solution" in text.lower():
            problem = JEEProblem(
                id=f"{source_file}_{page_num}_{len(problems)+1}",
                subject="mathematics",  # Would be auto-detected
                topic="unknown",       # Would be classified by AI
                difficulty="medium",   # Would be assessed by AI
                problem_text=text[:200] + "...",  # Extract actual problem
                solution="Solution extracted here...",  # Extract solution
                answer="Answer here",  # Extract final answer
                source_file=source_file,
                page_number=page_num,
                created_at=datetime.now().isoformat()
            )
            problems.append(problem)
        
        return problems
    
    def process_pdf_directory(self, pdf_dir: Path = None) -> Dict[str, Any]:
        """Process all PDFs in directory and extract problems"""
        if pdf_dir is None:
            pdf_dir = self.base_dir / "raw_pdfs"
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {pdf_dir}")
            print(f"   Please add JEE preparation PDFs to {pdf_dir}")
            return {"total_problems": 0, "processed_files": 0}
        
        all_problems = []
        processed_count = 0
        
        for pdf_path in pdf_files:
            print(f"\n🔄 Processing: {pdf_path.name}")
            
            # Parse PDF with LangChain
            documents = self.parse_pdf_with_langchain(pdf_path)
            
            # Extract problems from each page
            for i, doc in enumerate(documents):
                problems = self.extract_problems_from_text(
                    doc.page_content, 
                    pdf_path.name, 
                    i + 1
                )
                all_problems.extend(problems)
            
            processed_count += 1
            
            # Save progress after each file
            self.save_problems_to_db(all_problems)
            self.save_problems_to_json(all_problems)
        
        stats = {
            "total_problems": len(all_problems),
            "processed_files": processed_count,
            "timestamp": datetime.now().isoformat()
        }
        
        self.save_statistics(stats)
        return stats
    
    def save_problems_to_db(self, problems: List[JEEProblem]):
        """Save problems to local SQLite database"""
        if not problems:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for problem in problems:
                conn.execute("""
                    INSERT OR REPLACE INTO problems VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    problem.id, problem.subject, problem.topic, problem.difficulty,
                    problem.problem_text, problem.solution, problem.answer,
                    problem.source_file, problem.page_number, problem.created_at
                ))
        
        print(f"💾 Saved {len(problems)} problems to database")
    
    def save_problems_to_json(self, problems: List[JEEProblem]):
        """Save problems to structured JSON file"""
        json_path = self.base_dir / "structured" / "jee_problems.json"
        
        problems_data = [problem.dict() for problem in problems]
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(problems_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(problems)} problems to {json_path}")
    
    def save_statistics(self, stats: Dict[str, Any]):
        """Save dataset statistics"""
        stats_path = self.base_dir / "statistics" / "dataset_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Dataset statistics saved to {stats_path}")
    
    def query_problems(self, subject: str = None, difficulty: str = None, limit: int = 10) -> List[Dict]:
        """Query problems from local database"""
        query = "SELECT * FROM problems WHERE 1=1"
        params = []
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        
        query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def generate_sample_pdfs_guide(self):
        """Generate guide for obtaining sample PDFs"""
        guide_path = self.base_dir / "raw_pdfs" / "README.md"
        
        guide_content = """# JEE Math PDF Sources

Add your JEE preparation PDFs to this directory for dataset creation.

## Recommended Sources:
1. **NCERT Textbooks** (Class 11-12 Math)
2. **JEE Main/Advanced Previous Years** 
3. **Popular Prep Books:**
   - RD Sharma
   - HC Verma (Physics with math)
   - Cengage Mathematics
   - Arihant JEE Main/Advanced
4. **Mock Test Papers**
5. **Online Sources:**
   - Archive.org educational materials
   - Open courseware PDFs
   - MIT OpenCourseWare

## File Naming Convention:
- `ncert_class11_math.pdf`
- `jee_main_2023_paper1.pdf`
- `rd_sharma_algebra.pdf`

## Processing:
Run `python create_local_dataset.py` to extract problems from all PDFs.
"""
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📚 Sample PDF guide created: {guide_path}")

def main():
    """Main execution function"""
    print("🚀 JEE Math Local Dataset Creator")
    print("=" * 50)
    
    # Initialize dataset creator
    creator = LocalDatasetCreator()
    
    # Generate guide for users
    creator.generate_sample_pdfs_guide()
    
    # Process PDFs (if any exist)
    stats = creator.process_pdf_directory()
    
    print("\n" + "=" * 50)
    print("📊 DATASET CREATION SUMMARY")
    print(f"✅ Total problems extracted: {stats['total_problems']}")
    print(f"✅ Files processed: {stats['processed_files']}")
    
    if stats['total_problems'] > 0:
        # Show sample query
        print("\n🔍 Sample problems:")
        sample_problems = creator.query_problems(limit=3)
        for i, problem in enumerate(sample_problems, 1):
            print(f"{i}. {problem['problem_text'][:100]}...")
    else:
        print("\n💡 Next steps:")
        print("1. Add JEE preparation PDFs to data/raw_pdfs/")
        print("2. Run this script again to extract problems")
        print("3. Use the extracted dataset for model training")
    
    print("\n🎯 Local dataset ready for model training!")

if __name__ == "__main__":
    main()
