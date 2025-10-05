import json

# Load the dataset
with open('jee_math_dataset/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 Dataset Analysis - {len(data)} examples")
print("=" * 50)

# Show first few examples
print("\n🔍 Sample Examples:")
for i, example in enumerate(data[:3]):
    print(f"\nExample {i+1}:")
    print(f"  ID: {example['id']}")
    print(f"  Length: {example['text_length']} chars")
    print(f"  Has math symbols: {example['has_math_symbols']}")
    print(f"  Has math terms: {example['has_math_terms']}")
    print(f"  Text preview: {example['original_text'][:200]}...")

# Analyze quality metrics
lengths = [ex['text_length'] for ex in data]
with_symbols = sum(1 for ex in data if ex['has_math_symbols'])
with_terms = sum(1 for ex in data if ex['has_math_terms'])

print(f"\n📈 Quality Metrics:")
print(f"  • Average length: {sum(lengths)/len(lengths):.0f} characters")
print(f"  • Min length: {min(lengths)} characters")
print(f"  • Max length: {max(lengths)} characters")
print(f"  • With math symbols: {with_symbols} ({with_symbols/len(data)*100:.1f}%)")
print(f"  • With math terms: {with_terms} ({with_terms/len(data)*100:.1f}%)")

# Check for actual mathematical content
good_examples = 0
for ex in data:
    text = ex['original_text'].lower()
    if any(word in text for word in ['find', 'solve', 'calculate', 'prove', 'show']):
        good_examples += 1

print(f"  • With action words (find/solve/etc): {good_examples} ({good_examples/len(data)*100:.1f}%)")
