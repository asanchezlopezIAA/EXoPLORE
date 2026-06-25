"""
exoplore.analysis.utils
========================

Miscellaneous helper utilities for analysis workflows.
"""

import os, shutil

def remove_all_elements(folder_path):
    """Recursively delete all files and subdirectories inside a folder.

    Iterates over the contents of ``folder_path`` and removes each entry:
    files and symlinks are deleted with ``os.unlink``; subdirectories are
    removed with ``shutil.rmtree``.  The parent folder itself is preserved.
    Deletion failures are caught and printed rather than raised, so the
    function continues cleaning up remaining entries even if one fails.

    Parameters
    ----------
    folder_path : str
        Path to the directory to empty.  If the path does not exist, a
        message is printed and the function returns without error.
    """
    import os
    import shutil
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it's a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')


