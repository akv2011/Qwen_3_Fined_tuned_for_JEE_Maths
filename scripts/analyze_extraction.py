import json
import pprint

# Load the extraction results
with open('jee_sample_extraction.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("🔍 JEE PDF Extraction Analysis")
print("=" * 40)

# Show summary
summary = data['analysis_summary']
print(f"📊 Analysis Summary:")
print(f"   Pages analyzed: {summary['total_pages_analyzed']}")
print(f"   Problems detected: {summary['total_problems_detected']}")
print(f"   Tables detected: {summary['total_tables_detected']}")
print(f"   Math notation: {summary['total_math_notation']}")

print(f"\n📈 Detection Rates:")
print(f"   Problem detection: {summary['problem_detection_rate']:.1%}")
print(f"   Table detection: {summary['table_detection_rate']:.1%}")
print(f"   Math detection: {summary['math_detection_rate']:.1%}")

print(f"\n📄 Page-by-Page Breakdown:")
for page in data['sample_pages'][:10]:
    page_num = page['page_number']
    text_len = page.get('text_length', 0)
    problems = len(page.get('problems_detected', []))
    tables = page.get('table_count', 0)
    math = len(page.get('math_notation', []))
    
    print(f"   Page {page_num}: {text_len:4d} chars, {problems:2d} problems, {tables:2d} tables, {math:2d} math")

# Find pages with actual problems
print(f"\n🎯 Pages with Detected Problems:")
for page in data['sample_pages']:
    problems = page.get('problems_detected', [])
    if problems:
        print(f"\n   📖 Page {page['page_number']} ({len(problems)} problems):")
        for i, prob in enumerate(problems[:2]):  # Show first 2 problems
            question = prob.get('question_text', '')[:80]
            print(f"      {i+1}. {question}...")

# Show sample math notation
print(f"\n🔢 Sample Math Notation Detected:")
all_math = []
for page in data['sample_pages']:
    all_math.extend(page.get('math_notation', []))

unique_math = list(set(all_math))[:10]  # Show first 10 unique
for math in unique_math:
    print(f"   • {math}")

print(f"\n✅ Analysis complete!")
