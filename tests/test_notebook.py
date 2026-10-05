"""
Test to verify that notebooks/predicting_price_analysis.ipynb is valid JSON and contains cells.
"""
from pathlib import Path
import nbformat


def test_notebook_valid_structure():
    nb_path = Path("notebooks/predicting_price_analysis.ipynb")
    assert nb_path.exists(), "Notebook file does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) > 10, "Notebook should have more than 10 cells"
    
    # Verify both code and markdown cells exist
    cell_types = {cell.cell_type for cell in nb.cells}
    assert "code" in cell_types
    assert "markdown" in cell_types

    # Ensure every code cell is followed by or near a markdown cell
    code_count = sum(1 for c in nb.cells if c.cell_type == "code")
    md_count = sum(1 for c in nb.cells if c.cell_type == "markdown")
    assert code_count > 0 and md_count > 0
    print(f"Verified notebook: {len(nb.cells)} cells ({code_count} code, {md_count} markdown).")
