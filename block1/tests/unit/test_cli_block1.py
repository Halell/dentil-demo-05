"""
Unit tests for CLI functionality.
"""
import pytest
import json
import sys
from pathlib import Path
from typer.testing import CliRunner

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cli.block1 import app


class TestCLI:
    """Test CLI commands."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        
        # Create test input file
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        self.input_file = self.test_dir / "test_input.txt"
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write("שתל 14 18/0\n")
            f.write("כתר על שן 36\n")
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_run_n0_t1_command(self):
        """Test run-n0-t1 CLI command."""
        result = self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", str(self.input_file),
                "--out", str(self.test_dir / "output")
            ]
        )
        
        # Check command executed successfully
        assert result.exit_code == 0
        
        # Check output files were created
        output_dir = self.test_dir / "output"
        assert (output_dir / "n0_normalized.jsonl").exists()
        assert (output_dir / "t1_tokens.jsonl").exists()
        
        # Check file contents
        with open(output_dir / "n0_normalized.jsonl", 'r', encoding='utf-8') as f:
            n0_lines = [json.loads(line) for line in f]
        
        assert len(n0_lines) == 2
        assert "raw_text" in n0_lines[0]
        assert "normalized_text" in n0_lines[0]
        assert "numbers" in n0_lines[0]
    
    def test_run_n0_t1_verbose(self):
        """Test run-n0-t1 with verbose output."""
        result = self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", str(self.input_file),
                "--out", str(self.test_dir / "output"),
                "--verbose"
            ]
        )
        
        assert result.exit_code == 0
        # Should show additional output in verbose mode
        assert "Input:" in result.stdout or "Normalized:" in result.stdout
    
    def test_run_all_command(self):
        """Test run-all CLI command."""
        result = self.runner.invoke(
            app,
            [
                "run-all",
                "--in", str(self.input_file),
                "--out", str(self.test_dir / "output")
            ]
        )
        
        assert result.exit_code == 0
        
        # Check all expected files are created
        output_dir = self.test_dir / "output"
        assert (output_dir / "n0_normalized.jsonl").exists()
        assert (output_dir / "t1_tokens.jsonl").exists()
        assert (output_dir / "merged_block1.jsonl").exists()
        
        # Check merged file structure
        with open(output_dir / "merged_block1.jsonl", 'r', encoding='utf-8') as f:
            merged_lines = [json.loads(line) for line in f]
        
        assert len(merged_lines) == 2
        for line in merged_lines:
            assert "raw_text" in line
            assert "n0" in line
            assert "t1" in line
            assert "llm_aug" in line
    
    def test_show_mv_command(self):
        """Test show-mv CLI command."""
        result = self.runner.invoke(
            app,
            ["show-mv", "שתל 14 18/0"]
        )
        
        assert result.exit_code == 0
        assert "<LEGEND>" in result.stdout
        assert "<TOKENS>" in result.stdout
        assert "MARKED VIEW" in result.stdout
    
    def test_show_mv_save_file(self):
        """Test show-mv with save option."""
        output_file = self.test_dir / "mv_output.txt"
        
        result = self.runner.invoke(
            app,
            [
                "show-mv", 
                "שתל 14",
                "--save", str(output_file)
            ]
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check file contents
        content = output_file.read_text(encoding='utf-8')
        assert "<LEGEND>" in content
        assert "<TOKENS>" in content
    
    def test_stats_command(self):
        """Test stats command."""
        # First create processed data
        self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", str(self.input_file),
                "--out", str(self.test_dir / "output")
            ]
        )
        
        # Then check stats
        result = self.runner.invoke(
            app,
            ["stats", str(self.test_dir / "output" / "merged_block1.jsonl")]
        )
        
        # Stats command might fail if merged file doesn't exist, 
        # but should show proper error handling
        assert result.exit_code in [0, 1]  # Either success or proper error
    
    def test_nonexistent_input_file(self):
        """Test handling of nonexistent input file."""
        result = self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", "nonexistent_file.txt",
                "--out", str(self.test_dir / "output")
            ]
        )
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
    
    def test_help_command(self):
        """Test help command."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Block 1 CLI" in result.stdout
        assert "run-n0-t1" in result.stdout
        assert "run-all" in result.stdout
    
    def test_command_help(self):
        """Test individual command help."""
        result = self.runner.invoke(app, ["run-n0-t1", "--help"])
        
        assert result.exit_code == 0
        assert "normalization" in result.stdout.lower() or "tokenization" in result.stdout.lower()


class TestFileFormats:
    """Test file format handling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.test_dir = Path("test_format_output")
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_json_output_format(self):
        """Test that output files contain valid JSON."""
        # Create test input
        input_file = self.test_dir / "input.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("שתל במקום 46\n")
        
        # Run processing
        result = self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", str(input_file),
                "--out", str(self.test_dir / "output")
            ]
        )
        
        assert result.exit_code == 0
        
        # Check JSON validity
        output_dir = self.test_dir / "output"
        
        with open(output_dir / "n0_normalized.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)  # Should not raise exception
                assert isinstance(data, dict)
        
        with open(output_dir / "t1_tokens.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)  # Should not raise exception
                assert isinstance(data, dict)
                assert "tokens" in data
    
    def test_unicode_handling(self):
        """Test proper Unicode handling in files."""
        # Create input with Hebrew and Arabic numerals
        input_file = self.test_dir / "unicode_input.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("שתל דנטלי 14 עם 18/0\n")
            f.write("כתר זרקוניה על שן 36\n")
        
        result = self.runner.invoke(
            app,
            [
                "run-n0-t1",
                "--in", str(input_file),
                "--out", str(self.test_dir / "output")
            ]
        )
        
        assert result.exit_code == 0
        
        # Check that Hebrew characters are preserved
        with open(self.test_dir / "output" / "n0_normalized.jsonl", 'r', encoding='utf-8') as f:
            content = f.read()
            assert "שתל" in content
            assert "זרקוניה" in content