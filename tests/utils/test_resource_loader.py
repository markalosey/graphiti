import hashlib
import os
import unittest
from unittest.mock import patch, mock_open

from graphiti_core.utils.resource_loader import get_formatting_guidelines, GuidelineError

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
_VALID_GUIDELINES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    '..',
    'graphiti_core',
    'data',
    'formatting_guidelines.md',
)


class TestGetFormattingGuidelines(unittest.TestCase):
    def test_successful_load_and_hash(self):
        """Test successful loading of the actual guidelines file and MD5 hash calculation."""
        # Ensure the actual file can be read by the function as a basic integration check
        try:
            content, md5_hash = get_formatting_guidelines()
            self.assertIsInstance(content, str)
            self.assertTrue(len(content) > 0)
            self.assertIsInstance(md5_hash, str)
            self.assertEqual(len(md5_hash), 32)  # MD5 hashes are 32 hex characters

            # Calculate hash independently for verification
            expected_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            self.assertEqual(md5_hash, expected_hash)
        except GuidelineError as e:
            self.fail(f'get_formatting_guidelines raised GuidelineError unexpectedly: {e}')

    @patch(
        'graphiti_core.utils.resource_loader.open',
        new_callable=mock_open,
        read_data='Test guideline content.',
    )
    def test_mocked_successful_load(self, mock_file):
        """Test successful loading with a mocked file."""
        content, md5_hash = get_formatting_guidelines()
        self.assertEqual(content, 'Test guideline content.')
        expected_hash = hashlib.md5('Test guideline content.'.encode('utf-8')).hexdigest()
        self.assertEqual(md5_hash, expected_hash)
        # Check that the correct file path was attempted to be opened by the actual function
        # The path used in get_formatting_guidelines is relative to its own location.
        # Construct the expected path based on the mocked open's perspective if needed,
        # or ensure the function uses an absolute path construction that can be asserted.
        # For this test, we assume the internal path resolution is correct if it gets here.

    @patch(
        'graphiti_core.utils.resource_loader.open',
        side_effect=FileNotFoundError('File not found for test'),
    )
    def test_file_not_found(self, mock_open_call):
        """Test that GuidelineError is raised if the file is not found."""
        with self.assertRaisesRegex(GuidelineError, 'Formatting guidelines file not found at'):
            get_formatting_guidelines()

    @patch('graphiti_core.utils.resource_loader.open', side_effect=IOError('Read error for test'))
    def test_io_error_on_read(self, mock_open_call):
        """Test that GuidelineError is raised on an IOError during file reading."""
        with self.assertRaisesRegex(GuidelineError, 'Error reading formatting guidelines file at'):
            get_formatting_guidelines()

    def test_actual_file_exists(self):
        """Verify that the actual guidelines file exists at the expected path."""
        self.assertTrue(
            os.path.exists(_VALID_GUIDELINES_PATH),
            f'The guidelines file is missing at the expected location: {_VALID_GUIDELINES_PATH}',
        )


if __name__ == '__main__':
    unittest.main()
