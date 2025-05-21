import hashlib
import os
from typing import Tuple, Optional

# The path to the guidelines file relative to this file's location
_GUIDELINES_FILE_NAME = '../data/formatting_guidelines.md'
_GUIDELINES_FILE_PATH = os.path.join(os.path.dirname(__file__), _GUIDELINES_FILE_NAME)


class GuidelineError(Exception):
    """Custom exception for guideline loading errors."""

    pass


def get_formatting_guidelines() -> Tuple[str, str]:
    """
    Reads the formatting_guidelines.md file, calculates its MD5 hash,
    and returns the content and the hash.

    Returns:
        Tuple[str, str]: A tuple containing the file content (str) and its MD5 hash (str).

    Raises:
        GuidelineError: If the file cannot be found or read.
    """
    try:
        with open(_GUIDELINES_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return content, md5_hash
    except FileNotFoundError:
        raise GuidelineError(f'Formatting guidelines file not found at {_GUIDELINES_FILE_PATH}')
    except IOError as e:
        raise GuidelineError(
            f'Error reading formatting guidelines file at {_GUIDELINES_FILE_PATH}: {e}'
        )


if __name__ == '__main__':
    # Example usage:
    try:
        guidelines_content, guidelines_hash = get_formatting_guidelines()
        print('Successfully loaded formatting guidelines.')
        print(f'MD5 Hash: {guidelines_hash}')
        print('\nFirst 200 characters of content:\n---')
        print(guidelines_content[:200])
        print('---')
    except GuidelineError as e:
        print(f'Error: {e}')
