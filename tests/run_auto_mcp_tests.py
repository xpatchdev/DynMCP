#!/usr/bin/env python3
"""
run_auto_mcp_tests.py - Simple test runner for AutoMCP

This script provides an easy way to run various AutoMCP tests.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def main():
    """Main test runner."""
    print("AutoMCP Test Runner")
    print("="*60)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tests = [
        ("python3 test_auto_mcp.py", "Basic AutoMCP Test Suite"),
        ("python3 test_auto_mcp.py --interactive", "Interactive AutoMCP Test"),
    ]
    
    print("\nAvailable tests:")
    for i, (cmd, desc) in enumerate(tests, 1):
        print(f"  {i}. {desc}")
    
    print("\nWhat would you like to run?")
    print("  1. Run basic test suite")
    print("  2. Run interactive test")
    print("  3. Start test server (you'll need to Ctrl+C to stop)")
    print("  4. Run all tests")
    print("  q. Quit")
    
    while True:
        choice = input("\nEnter your choice (1-4, q): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == '1':
            run_command("python3 test_auto_mcp.py", "Basic AutoMCP Test Suite")
        elif choice == '2':
            run_command("python3 test_auto_mcp.py --interactive", "Interactive AutoMCP Test")
        elif choice == '3':
            print("\nStarting test server...")
            print("The server will run on http://localhost:8000")
            print("Press Ctrl+C to stop the server")
            try:
                subprocess.run(["python3", "test_auto_mcp.py", "--run-server"], check=False)
            except KeyboardInterrupt:
                print("\nServer stopped.")
        elif choice == '4':
            print("\nRunning all tests...")
            run_command("python3 test_auto_mcp.py", "Basic AutoMCP Test Suite")
            run_command("python3 test_auto_mcp.py --interactive", "Interactive AutoMCP Test")
        else:
            print("Invalid choice. Please enter 1-4 or q.")

if __name__ == "__main__":
    main()
