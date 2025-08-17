import subprocess


def run_bash_script(script_path: str, *args) -> str:
    """
    Executes a Bash script file and returns the output.

    Args:
        script_path (str): Path to the Bash script file.
        *args: Additional arguments to pass to the script.

    Returns:
        str: Standard output of the executed script.
    """
    try:
        # Construct the command with arguments
        command = ["bash", script_path] + list(args)

        # Execute the script and capture the output
        result = subprocess.run(command, text=True, capture_output=True, check=True)

        return result.stdout  # Return the standard output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"  # Return the error message if execution fails


def run_bash_command(command: str) -> str:
    """
    Executes a shell command and returns the output.
    Uses PowerShell on Windows, bash on Unix-like systems.

    Args:
        command (str): The shell command to execute.

    Returns:
        str: The standard output of the executed command.
    """
    import platform
    
    try:
        # Use appropriate shell based on platform
        if platform.system() == "Windows":
            # Use PowerShell on Windows
            result = subprocess.run(
                ["powershell", "-Command", command], 
                text=True, capture_output=True, check=True
            )
        else:
            # Use bash on Unix-like systems
            result = subprocess.run(
                command, shell=True, text=True, capture_output=True, check=True
            )

        return result.stdout  # Return the standard output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"  # Return the error message if execution fails
