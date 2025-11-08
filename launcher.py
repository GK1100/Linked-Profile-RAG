#!/usr/bin/env python3
"""
LinkedIn Profile Scraping & RAG System Launcher
===============================================

This launcher provides options to:
1. Scrape LinkedIn profiles only
2. Start RAG application only (with existing data)
3. Run complete workflow (scrape + RAG)
"""

import sys
import subprocess
import os

def print_banner():
    """Print the system banner"""
    print("🎯 LinkedIn Profile Scraping & RAG System")
    print("=" * 50)
    print("A comprehensive system for scraping LinkedIn profiles")
    print("and querying them using Retrieval-Augmented Generation")
    print("=" * 50)

def run_scraping_only():
    """Run only the LinkedIn scraping functionality"""
    print("🔍 Running LinkedIn Profile Scraping...")
    try:
        subprocess.run([sys.executable, "ls3.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Scraping stopped by user")
    except Exception as e:
        print(f"❌ Error during scraping: {e}")

def run_rag_only():
    """Run only the RAG application"""
    print("🚀 Starting RAG Web Application...")
    print("🌐 The web interface will be available at: http://localhost:5000")
    try:
        subprocess.run([sys.executable, "linkedin_rag_webapp.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 RAG application stopped by user")
    except Exception as e:
        print(f"❌ Error starting RAG application: {e}")

def run_complete_workflow():
    """Run the complete integrated workflow"""
    print("🔄 Running Complete Workflow...")
    try:
        subprocess.run([sys.executable, "integrated_linkedin_system.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Workflow stopped by user")
    except Exception as e:
        print(f"❌ Error in workflow: {e}")

def check_data_file():
    """Check if data file exists and show info"""
    data_file = "linkedin_profiless_ls3.json"
    if os.path.exists(data_file):
        try:
            import json
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📊 Found existing data: {len(data)} profiles")
            return True
        except:
            print("⚠️ Data file exists but may be corrupted")
            return False
    else:
        print("📂 No existing data file found")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check existing data
    has_data = check_data_file()
    print()
    
    while True:
        print("📋 Choose an option:")
        print("1. 🔍 Scrape LinkedIn Profiles Only")
        print("2. 🚀 Start RAG Application Only")
        print("3. 🔄 Run Complete Workflow (Scrape + RAG)")
        print("4. 📊 Check Data Status")
        print("5. ❌ Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print()
                run_scraping_only()
                print()
                
            elif choice == "2":
                if not has_data:
                    print("⚠️ No data file found. Please scrape profiles first (option 1)")
                    print()
                else:
                    print()
                    run_rag_only()
                    print()
                    
            elif choice == "3":
                print()
                run_complete_workflow()
                print()
                
            elif choice == "4":
                print()
                has_data = check_data_file()
                print()
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    main()
