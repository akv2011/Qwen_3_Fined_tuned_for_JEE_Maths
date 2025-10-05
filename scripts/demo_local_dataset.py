#!/usr/bin/env python3
"""
Demo: Local JEE Math Dataset Creation
Quick demonstration of the local approach using sample data
"""

import json
from pathlib import Path
from create_local_dataset import LocalDatasetCreator, JEEProblem

def create_sample_data():
    """Create sample data to demonstrate the local dataset approach"""
    creator = LocalDatasetCreator("demo_data")
    
    # Create sample problems (simulating PDF extraction)
    sample_problems = [
        JEEProblem(
            id="sample_001",
            subject="algebra", 
            topic="quadratic_equations",
            difficulty="medium",
            problem_text="Find the roots of the quadratic equation x² - 5x + 6 = 0",
            solution="Using the quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2",
            answer="x = 3, x = 2",
            source_file="sample_algebra.pdf",
            page_number=15,
            created_at="2025-08-13T17:45:00"
        ),
        JEEProblem(
            id="sample_002",
            subject="calculus",
            topic="derivatives", 
            difficulty="hard",
            problem_text="Find the derivative of f(x) = x³ sin(2x)",
            solution="Using product rule: f'(x) = 3x²sin(2x) + x³·2cos(2x) = 3x²sin(2x) + 2x³cos(2x)",
            answer="f'(x) = 3x²sin(2x) + 2x³cos(2x)",
            source_file="sample_calculus.pdf", 
            page_number=42,
            created_at="2025-08-13T17:45:30"
        ),
        JEEProblem(
            id="sample_003",
            subject="geometry",
            topic="coordinate_geometry",
            difficulty="easy", 
            problem_text="Find the distance between points A(1,2) and B(4,6)",
            solution="Using distance formula: d = √[(4-1)² + (6-2)²] = √[9 + 16] = √25 = 5",
            answer="5 units",
            source_file="sample_geometry.pdf",
            page_number=8,
            created_at="2025-08-13T17:46:00"
        )
    ]
    
    print("🔧 Creating sample JEE math dataset...")
    
    # Save to database and JSON
    creator.save_problems_to_db(sample_problems)
    creator.save_problems_to_json(sample_problems)
    
    # Create statistics
    stats = {
        "total_problems": len(sample_problems),
        "subjects": {"algebra": 1, "calculus": 1, "geometry": 1},
        "difficulties": {"easy": 1, "medium": 1, "hard": 1},
        "processed_files": 3,
        "timestamp": "2025-08-13T17:46:00"
    }
    creator.save_statistics(stats)
    
    return creator, sample_problems

def demonstrate_queries(creator):
    """Demonstrate local database queries"""
    print("\n🔍 Demonstrating local database queries:")
    print("-" * 40)
    
    # Query by subject
    print("📚 Algebra problems:")
    algebra_problems = creator.query_problems(subject="algebra", limit=5)
    for problem in algebra_problems:
        print(f"  • {problem['problem_text'][:60]}...")
    
    print("\n📊 Calculus problems:")
    calculus_problems = creator.query_problems(subject="calculus", limit=5)
    for problem in calculus_problems:
        print(f"  • {problem['problem_text'][:60]}...")
    
    print("\n🎯 Hard difficulty problems:")
    hard_problems = creator.query_problems(difficulty="hard", limit=5)
    for problem in hard_problems:
        print(f"  • {problem['problem_text'][:60]}...")

def show_data_structure(creator):
    """Show the local data structure created"""
    print("\n📁 Local Data Structure:")
    print("-" * 30)
    
    base_path = Path("demo_data")
    for item in base_path.rglob("*"):
        if item.is_file():
            size = item.stat().st_size
            print(f"📄 {item.relative_to(base_path)} ({size} bytes)")
        elif item.is_dir():
            print(f"📁 {item.relative_to(base_path)}/")

def show_benefits():
    """Show benefits of local approach"""
    print("\n✅ Benefits of Local Dataset Approach:")
    print("-" * 40)
    benefits = [
        "💰 No cloud storage costs",
        "🚀 Faster iteration and debugging", 
        "🔒 Full control over data processing",
        "📶 Works completely offline",
        "🔍 Easy data inspection and validation",
        "⚡ Fast local SQLite queries",
        "🛠️ Simple setup with LangChain",
        "📊 Local statistics and reporting",
        "🔄 Easy to modify and extend",
        "🎯 Perfect for model training"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")

def main():
    """Run the complete demonstration"""
    print("🎯 JEE Math Local Dataset Demo")
    print("=" * 50)
    
    # Create sample dataset
    creator, problems = create_sample_data()
    
    # Show data structure
    show_data_structure(creator)
    
    # Demonstrate queries
    demonstrate_queries(creator)
    
    # Show benefits
    show_benefits()
    
    print("\n" + "=" * 50)
    print("📋 Next Steps for Real Implementation:")
    print("1. Install requirements: pip install -r requirements_local.txt")
    print("2. Add JEE PDFs to data/raw_pdfs/ directory")
    print("3. Run: python create_local_dataset.py")
    print("4. Use extracted dataset for model training")
    print("5. No cloud setup needed - everything local!")
    
    print("\n🎉 Local dataset approach ready for JEE math model training!")

if __name__ == "__main__":
    main()
