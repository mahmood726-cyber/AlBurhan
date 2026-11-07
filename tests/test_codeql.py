"""CodeQL jobs must target languages present in this repository."""
import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_codeql_languages_have_source_files():
    workflow = (REPO_ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")
    matrix = next(line.split("language:", 1)[1].strip()
                  for line in workflow.splitlines() if "language:" in line)
    extensions = {"python": (".py",), "javascript": (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")}
    for language in ast.literal_eval(matrix):
        assert any(path.is_file() and path.suffix in extensions[language]
                   for path in (REPO_ROOT / "alburhan").rglob("*")), language
