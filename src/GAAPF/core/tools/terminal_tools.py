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
    Executes a Bash command and returns the output.

    Args:
        command (str): The Bash command to execute.

    Returns:
        str: The standard output of the executed command.
    """
    try:
        # Execute the command
        result = subprocess.run(
            command, shell=True, text=True, capture_output=True, check=True
        )

        return result.stdout  # Return the standard output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"  # Return the error message if execution fails
