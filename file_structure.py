#!/usr/bin/env python3
"""
Project File Structure Fetcher
Generates a tree-like visualization of your project's directory structure.
"""

import os
import argparse
from pathlib import Path

def should_ignore(path, ignore_patterns):
    """Check if a path should be ignored based on patterns."""
    path_str = str(path)
    name = path.name
    
    for pattern in ignore_patterns:
        if pattern in path_str or pattern == name or path_str.endswith(pattern):
            return True
    return False

def get_file_structure(root_path, ignore_patterns=None, max_depth=None):
    """
    Get the file structure of a directory.
    
    Args:
        root_path: Path to the root directory
        ignore_patterns: List of patterns to ignore
        max_depth: Maximum depth to traverse (None for unlimited)
    
    Returns:
        List of tuples (path, depth, is_file)
    """
    if ignore_patterns is None:
        ignore_patterns = {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules', 
            '.vscode', '.idea', '*.pyc', '.DS_Store', 'venv', 
            'env', '.env', 'dist', 'build', '*.egg-info'
        }
    
    root = Path(root_path)
    structure = []
    
    def traverse(path, depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        if should_ignore(path, ignore_patterns):
            return
            
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                if should_ignore(item, ignore_patterns):
                    continue
                    
                structure.append((item, depth, item.is_file()))
                
                if item.is_dir():
                    traverse(item, depth + 1)
                    
        except PermissionError:
            pass  # Skip directories we can't read
    
    structure.append((root, 0, False))
    traverse(root)
    
    return structure

def format_tree(structure, show_files=True):
    """Format the structure as a tree."""
    lines = []
    
    for i, (path, depth, is_file) in enumerate(structure):
        if not show_files and is_file:
            continue
            
        # Create the tree symbols
        prefix = "│   " * depth
        
        if depth > 0:
            # Check if this is the last item at this depth
            is_last = True
            for j in range(i + 1, len(structure)):
                if structure[j][1] < depth:
                    break
                elif structure[j][1] == depth:
                    is_last = False
                    break
            
            if is_last:
                prefix = prefix[:-4] + "└── "
            else:
                prefix = prefix[:-4] + "├── "
        
        # Add file/folder indicator
        if is_file:
            icon = "📄 "
        else:
            icon = "📁 " if depth > 0 else "📂 "
        
        lines.append(f"{prefix}{icon}{path.name}")
    
    return "\n".join(lines)

def save_to_file(content, output_file):
    """Save the structure to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"File structure saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate project file structure")
    parser.add_argument("path", nargs="?", default=".", 
                       help="Path to the project directory (default: current directory)")
    parser.add_argument("-o", "--output", help="Output file to save the structure")
    parser.add_argument("-d", "--max-depth", type=int, 
                       help="Maximum depth to traverse")
    parser.add_argument("--no-files", action="store_true", 
                       help="Show only directories, not files")
    parser.add_argument("--ignore", nargs="+", 
                       help="Additional patterns to ignore")
    
    args = parser.parse_args()
    
    # Validate path
    project_path = Path(args.path)
    if not project_path.exists():
        print(f"Error: Path '{args.path}' does not exist")
        return
    
    if not project_path.is_dir():
        print(f"Error: Path '{args.path}' is not a directory")
        return
    
    # Set up ignore patterns
    ignore_patterns = {
        '__pycache__', '.git', '.svn', '.hg', 'node_modules', 
        '.vscode', '.idea', '*.pyc', '.DS_Store', 'venv', 
        'env', '.env', 'dist', 'build', '*.egg-info'
    }
    
    if args.ignore:
        ignore_patterns.update(args.ignore)
    
    print(f"Scanning project: {project_path.absolute()}")
    print("-" * 50)
    
    # Get file structure
    structure = get_file_structure(
        project_path, 
        ignore_patterns=ignore_patterns,
        max_depth=args.max_depth
    )
    
    # Format as tree
    tree_output = format_tree(structure, show_files=not args.no_files)
    
    # Display results
    print(tree_output)
    print("-" * 50)
    print(f"Total items: {len(structure)}")
    
    # Save to file if requested
    if args.output:
        header = f"File Structure for: {project_path.absolute()}\n"
        header += f"Generated on: {os.popen('date').read().strip()}\n"
        header += "-" * 50 + "\n\n"
        
        full_content = header + tree_output
        save_to_file(full_content, args.output)

if __name__ == "__main__":
    main()