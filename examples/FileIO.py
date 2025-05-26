"""
File Input/Output Operations

This module provides file input/output operations: listing directory contents,
reading file contents with optional line range, and writing data to files with
optional line replacement. Directory and file paths are relative to the root
directory. Line numbers are 1-based and inclusive.
"""

import os

class FileIO:
    _root_dir = os.path.abspath(os.getcwd())

    @classmethod
    def _resolve_path(cls, path):
        # Join with root and resolve
        abs_path = os.path.abspath(os.path.join(cls._root_dir, path))
        # Ensure abs_path is within root_dir
        if not abs_path.startswith(cls._root_dir):
            raise PermissionError("Access denied: cannot go above the root directory.")
        return abs_path

    @staticmethod
    def ls(path=None):
        """List the contents of a directory. Uses current working directory if no path is provided."""
        if path is None:
            abs_path = FileIO._root_dir
        else:
            abs_path = FileIO._resolve_path(path)
        return os.listdir(abs_path)

    @staticmethod
    def read(path, from_line=None, to_line=None):
        """Read file contents, optionally with line range (1-based inclusive)."""
        abs_path = FileIO._resolve_path(path)
        with open(abs_path, 'r') as f:
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
    def write(path, data, from_line=None, to_line=None, create_dirs=True):
        """Write data to a file, optionally replacing specific lines (1-based inclusive)."""
        abs_path = FileIO._resolve_path(path)
        dir_name = os.path.dirname(abs_path)
        if create_dirs and dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if from_line is not None and to_line is not None:
            # Read all lines, replace the specified range, and write back
            with open(abs_path, 'r') as f:
                lines = f.readlines()
            # Convert 1-based to 0-based
            lines[from_line-1:to_line] = [data]
            with open(abs_path, 'w') as f:
                f.writelines(lines)
        elif from_line is not None:
            with open(abs_path, 'r') as f:
                lines = f.readlines()
            lines[from_line-1:] = [data]
            with open(abs_path, 'w') as f:
                f.writelines(lines)
        elif to_line is not None:
            with open(abs_path, 'r') as f:
                lines = f.readlines()
            lines[:to_line] = [data]
            with open(abs_path, 'w') as f:
                f.writelines(lines)
        else:
            with open(abs_path, 'w') as f:
                f.write(data)

    @staticmethod
    def cd(path):
        """Change the current working directory for subsequent file operations."""
        abs_path = FileIO._resolve_path(path)
        if not os.path.isdir(abs_path):
            raise NotADirectoryError(f"{abs_path} is not a directory.")
        FileIO._root_dir = abs_path
        return f"Current working directory set to: {abs_path}"