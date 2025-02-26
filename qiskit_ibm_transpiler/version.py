import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Union

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_command(cmd):
    """
    Run a shell command with a minimal environment and return its output.
    Args:
        cmd (list): Command to execute as a list of strings.
    Returns:
        str: The command's output.
    Raises:
        OSError: If the command execution fails.
    """

    env = {
        key: os.environ.get(key)
        for key in ["SYSTEMROOT", "PATH"]
        if os.environ.get(key)
    }
    env.update({"LANGUAGE": "C", "LANG": "C", "LC_ALL": "C"})

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(ROOT_DIR),
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode > 0:
                raise OSError(
                    f"Command {cmd} failed with code {proc.returncode}: {stderr.strip().decode('ascii')}"
                )
        return stdout.strip().decode("ascii")

    except Exception as e:
        raise OSError(f"Error running command {cmd}: {str(e)}")


def _get_version_from_metadata():
    """
    Retrieve the package version from importlib.metadata.
    Returns:
        str: The installed package version, or None if not found.
    """
    try:
        return version("qiskit-ibm-transpiler")
    except PackageNotFoundError:
        return "Unknown"


def _get_git_commit_hash() -> Union[str, None]:
    """
    Get the latest Git commit hash.
    Returns:
        str | None: The first 7 characters of the Git commit hash, or None if unavailable.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("ascii")[:7]
        )
    except subprocess.SubprocessError:
        return None


def _is_release_tag_present() -> bool:
    """
    Check if the current HEAD is pointing to a release tag.
    Returns:
        bool: True if a release tag is found, otherwise False.
    """
    try:
        release_tags = _run_command(["git", "tag", "-l", "--points-at", "HEAD"])
        return bool(release_tags)
    except OSError:
        return False


def get_version_info() -> str:
    """
    Retrieve the full version string:
    - If the package is installed, use importlib.metadata.version() (PEP standard)
      (https://docs.python.org/3/library/importlib.metadata.html)
    - If the repository is a Git repo but does not have a release tag, append the commit hash.
    Returns:
        str: The version string.
    """
    version_info = _get_version_from_metadata()
    is_a_git_repo = os.path.exists(os.path.join(ROOT_DIR, ".git"))

    if is_a_git_repo and not _is_release_tag_present():

        if git_hash := _get_git_commit_hash():
            version_info += f".dev0+{git_hash}"

    return version_info


__version__ = get_version_info()
