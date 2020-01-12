import os

def check_dir(DIR):
    """
    Check if directory already exists.
    If not, create the directory.
    """
    if not os.path.exists(DIR):
        print("Creating directory '%s' ...", (DIR))
        os.makedirs(DIR)
