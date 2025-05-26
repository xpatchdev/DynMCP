"""
File Input/Output Operations

This module provides file input/output operations: listing directory contents,
reading file contents with optional line range, and writing data to files with
optional line replacement. Directory and file paths are relative to the root
directory. Line numbers are 1-based and inclusive.
"""

import os

class FileIO:
    @staticmethod
    def list_directory(path="."):
        """List the contents of a directory."""
        return os.listdir(path)

    @staticmethod
    def read_file(path, from_line=None, to_line=None):
        """Read file contents, optionally with line range (1-based inclusive)."""
        with open(path, 'r') as f:
            lines = f.readlines()
        if from_line is not None and to_line is not None:
            # Convert 1-based to 0-based index
            return ''.join(lines[from_line-1:to_line])
        elif from_line is not None:
            return ''.join(lines[from_line-1:])
        elif to_line is not None:
            return ''.join(lines[:to_line])
        else:
            return ''.join(lines)

    @staticmethod
    def write_file(path, data, from_line=None, to_line=None, create_dirs=True):
        """Write data to a file, optionally replacing specific lines (1-based inclusive)."""
        dir_name = os.path.dirname(path)
        if create_dirs and dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if from_line is not None and to_line is not None:
            # Read all lines, replace the specified range, and write back
            with open(path, 'r') as f:
                lines = f.readlines()
            # Convert 1-based to 0-based
            lines[from_line-1:to_line] = [data]
            with open(path, 'w') as f:
                f.writelines(lines)
        elif from_line is not None:
            with open(path, 'r') as f:
                lines = f.readlines()
            lines[from_line-1:] = [data]
            with open(path, 'w') as f:
                f.writelines(lines)
        elif to_line is not None:
            with open(path, 'r') as f:
                lines = f.readlines()
            lines[:to_line] = [data]
            with open(path, 'w') as f:
                f.writelines(lines)
        else:
            with open(path, 'w') as f:
                f.write(data)