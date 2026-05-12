import os
from datetime import datetime


def get_folder_path(folder_path:str=" ", levels_up:int=0):
    """
    Returns the path to the specified folder or the path after moving up
    a specified number of directory levels without changing the current working directory.

    :param folder_path: A relative or absolute path to the target folder. If None, only levels_up will be applied.
    :param levels_up: The number of directory levels to go up. Default is 0 (no change).
    :return: The calculated path as a string.

    Example usage:
    Get the path to a specific folder:
        path = get_folder_path('your_folder_name')
    Get the path after moving up 2 levels:
        path = get_folder_path(levels_up=2)
    Get the path after moving up 1 level, then navigating to a folder:
        path = get_folder_path('your_folder_name', levels_up=1)
    Get the path to a subfolder within a folder:
        path = get_folder_path('parent_folder/sub_folder')
    """
    try:
        # Get the current working directory
        current_dir = os.getcwd()

        # If levels_up is specified, go up the directory tree
        if levels_up > 0:
            for _ in range(levels_up):
                # Move up one level by getting the parent directory (dirname of current_dir)
                current_dir = os.path.dirname(current_dir)

        # If a folder_path is specified, navigate to that folder
        if folder_path:
            # Handle multiple folder paths
            new_path = os.path.join(current_dir, folder_path)
        else:
            new_path = current_dir

        # Normalize the path to handle cases like '../' or './'
        new_path = os.path.normpath(new_path)

        return new_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def unix_time_to_datetime(unix_time):
    """
    Helper function to convert a Unix timestamp to a datetime object.
    """
    # Convert string to float
    unix_timestamp = float(unix_time)
    return datetime.utcfromtimestamp(unix_timestamp)
