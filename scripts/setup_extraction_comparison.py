"""
Setup script for JEE PDF Extraction Comparison
This script helps you install dependencies and set up API keys
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and print the result"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"✗ {description} failed")
            print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements file exists
    req_file = Path("requirements_comparison.txt")
    if not req_file.exists():
        print("✗ requirements_comparison.txt not found")
        return False
    
    # Install base requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements_comparison.txt",
        "Installing PDF extraction libraries"
    )
    
    if success:
        print("✓ All dependencies installed successfully")
    else:
        print("⚠️ Some dependencies may have failed to install")
        print("You can try installing them individually:")
        print("  pip install google-langextract")
        print("  pip install llama-parse llama-index")
        print("  pip install unstructured[pdf]")
    
    return success

def setup_api_keys():
    """Help user set up API keys"""
    print("\n🔑 Setting up API keys...")
    
    # Check for existing .env file
    env_file = Path(".env")
    env_content = []
    
    if env_file.exists():
        print("Found existing .env file")
        with open(env_file, 'r') as f:
            env_content = f.readlines()
    
    # Check current environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
    
    updates_needed = False
    
    # Google API Key for LangExtract
    if not google_api_key:
        print("\n🔍 Google API Key (for LangExtract) not found")
        print("To get a Google API Key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable the Generative AI API")
        print("4. Go to 'Credentials' and create an API key")
        
        key = input("\nEnter your Google API Key (or press Enter to skip): ").strip()
        if key:
            # Add to .env content
            env_content.append(f"GOOGLE_API_KEY={key}\n")
            updates_needed = True
            print("✓ Google API Key will be saved")
        else:
            print("⚠️ LangExtract will be skipped without Google API Key")
    else:
        print("✓ Google API Key found")
    
    # LlamaCloud API Key for LlamaParse
    if not llama_api_key:
        print("\n🦙 LlamaCloud API Key (for LlamaParse) not found")
        print("To get a LlamaCloud API Key:")
        print("1. Go to https://cloud.llamaindex.ai/")
        print("2. Sign up or log in")
        print("3. Navigate to API Keys section")
        print("4. Create a new API key")
        
        key = input("\nEnter your LlamaCloud API Key (or press Enter to skip): ").strip()
        if key:
            # Add to .env content
            env_content.append(f"LLAMA_CLOUD_API_KEY={key}\n")
            updates_needed = True
            print("✓ LlamaCloud API Key will be saved")
        else:
            print("⚠️ LlamaParse will be skipped without LlamaCloud API Key")
    else:
        print("✓ LlamaCloud API Key found")
    
    # Save .env file if updates needed
    if updates_needed:
        with open(env_file, 'w') as f:
            f.writelines(env_content)
        print(f"✓ API keys saved to {env_file}")
        print("Note: Restart your environment to load these keys")
    
    return True

def verify_installation():
    """Verify that libraries can be imported"""
    print("\n🧪 Verifying installation...")
    
    libraries = [
        ("PyMuPDF", "fitz"),
        ("Google LangExtract", "langextract"),
        ("LlamaParse", "llama_parse"),
        ("Unstructured", "unstructured"),
        ("Pandas", "pandas")
    ]
    
    results = []
    for name, module in libraries:
        try:
            __import__(module)
            print(f"✓ {name} - OK")
            results.append(True)
        except ImportError as e:
            print(f"✗ {name} - Failed: {e}")
            results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 Verification Results: {success_count}/{total_count} libraries available")
    
    if success_count >= 3:  # At least PyMuPDF, pandas, and one other
        print("✓ Sufficient libraries for comparison")
        return True
    else:
        print("⚠️ Some libraries missing - comparison will have limited results")
        return False

def create_sample_test():
    """Create a sample test script"""
    print("\n📝 Creating sample test script...")
    
    test_script = '''"""
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
    
    print("\\n" + comparison.generate_comparison_report())
    comparison.save_results()

if __name__ == "__main__":
    print("JEE PDF Extraction Test")
    print("Make sure to update the pdf_path variable with your actual PDF file!")
    
    # Uncomment the next line to run the test
    # asyncio.run(test_comparison())
'''
    
    test_file = Path("test_extraction.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✓ Sample test script created: {test_file}")
    print("Edit the pdf_path variable and uncomment the last line to run")

def main():
    """Main setup function"""
    print("🚀 JEE PDF Extraction Comparison Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Setup API keys
    setup_api_keys()
    
    # Verify installation
    verify_installation()
    
    # Create sample test
    create_sample_test()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Make sure your .env file has the API keys")
    print("2. Update test_extraction.py with your PDF path")
    print("3. Run: python jee_extraction_comparison.py your_pdf_file.pdf")
    print("\nOr run the async version:")
    print("python test_extraction.py")

if __name__ == "__main__":
    main()
