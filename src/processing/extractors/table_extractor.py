"""
Table Extractor

Extracts tables from PDF documents using Camelot and PDFPlumber.
Multi-tool approach for robustness.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import io

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

import pdfplumber
import pandas as pd

from ...utils.logger import logger


class TableExtractor:
    """
    Extracts tables from PDF documents

    Uses multiple extraction methods:
    1. Camelot (lattice mode for bordered tables)
    2. Camelot (stream mode for borderless tables)
    3. PDFPlumber (fallback)
    """

    def __init__(self, min_accuracy: float = 50.0):
        """
        Initialize table extractor

        Args:
            min_accuracy: Minimum accuracy threshold for Camelot tables (0-100)
        """
        self.min_accuracy = min_accuracy

        if not CAMELOT_AVAILABLE:
            logger.warning("Camelot not available, using PDFPlumber only")

    def extract_with_camelot(
        self,
        pdf_path: Path,
        pages: str = "all",
        flavor: str = "lattice"
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot

        Args:
            pdf_path: Path to PDF file
            pages: Page numbers to process (e.g., "1-3,5")
            flavor: 'lattice' for bordered tables, 'stream' for borderless

        Returns:
            List of extracted tables with metadata
        """
        if not CAMELOT_AVAILABLE:
            return []

        tables_data = []

        try:
            # Extract tables
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor=flavor,
                suppress_stdout=True
            )

            logger.debug(f"Camelot ({flavor}) found {len(tables)} tables")

            for i, table in enumerate(tables):
                # Check accuracy
                accuracy = table.accuracy if hasattr(table, 'accuracy') else 100.0

                if accuracy < self.min_accuracy:
                    logger.debug(f"Skipping table {i} (accuracy: {accuracy:.1f}%)")
                    continue

                # Convert to markdown format
                df = table.df
                markdown = self._dataframe_to_markdown(df)

                tables_data.append({
                    "table_num": i,
                    "page": table.page,
                    "accuracy": accuracy,
                    "markdown": markdown,
                    "dataframe": df,
                    "extractor": f"camelot_{flavor}",
                    "shape": table.shape
                })

        except Exception as e:
            logger.error(f"Camelot ({flavor}) extraction failed: {e}")

        return tables_data

    def extract_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using PDFPlumber

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers (0-indexed), None for all pages

        Returns:
            List of extracted tables with metadata
        """
        tables_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = pages if pages else range(len(pdf.pages))

                for page_num in pages_to_process:
                    if page_num >= len(pdf.pages):
                        continue

                    page = pdf.pages[page_num]

                    # Extract tables from page
                    page_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "explicit_vertical_lines": [],
                            "explicit_horizontal_lines": [],
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                            "min_words_vertical": 3,
                            "min_words_horizontal": 1,
                        }
                    )

                    for i, table in enumerate(page_tables):
                        if not table or len(table) == 0:
                            continue

                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])

                        # Convert to markdown
                        markdown = self._dataframe_to_markdown(df)

                        tables_data.append({
                            "table_num": len(tables_data),
                            "page": page_num + 1,  # 1-indexed for consistency
                            "markdown": markdown,
                            "dataframe": df,
                            "extractor": "pdfplumber",
                            "shape": df.shape
                        })

        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")

        return tables_data
    
    def _is_valid_table(self, df: pd.DataFrame) -> bool:
        """Filter out noise/junk tables"""
        # 1. Size Check: Single cells are rarely tables
        if df.shape[0] < 2 or df.shape[1] < 2:
            return False
            
        # 2. Empty Check: If > 50% of cells are empty strings, it's likely layout noise
        total_cells = df.size
        empty_cells = (df == "").sum().sum() + (df.isna()).sum().sum()
        if (empty_cells / total_cells) > 0.5:
            return False
            
        return True

    # Usage in extract_from_pdf:
    # all_tables = [t for t in all_tables if self._is_valid_table(t['dataframe'])]

    def extract_from_pdf(
        self,
        pdf_path: Path,
        use_camelot: bool = True,
        use_pdfplumber: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using multi-tool approach

        Args:
            pdf_path: Path to PDF file
            use_camelot: Whether to use Camelot
            use_pdfplumber: Whether to use PDFPlumber

        Returns:
            List of extracted tables, deduplicated
        """
        logger.debug(f"Extracting tables from {pdf_path.name}")

        all_tables = []

        # Try Camelot lattice (bordered tables)
        if use_camelot and CAMELOT_AVAILABLE:
            lattice_tables = self.extract_with_camelot(pdf_path, flavor="lattice")
            all_tables.extend(lattice_tables)

            # If few tables found, try stream mode
            if len(lattice_tables) < 2:
                stream_tables = self.extract_with_camelot(pdf_path, flavor="stream")
                all_tables.extend(stream_tables)

        # Try PDFPlumber
        if use_pdfplumber:
            plumber_tables = self.extract_with_pdfplumber(pdf_path)
            all_tables.extend(plumber_tables)

        # Deduplicate tables (same page + similar shape)
        deduplicated = self._deduplicate_tables(all_tables)

        logger.debug(f"Extracted {len(deduplicated)} unique tables")

        return deduplicated

    def _deduplicate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 1. Sort by accuracy so we keep the best version of a duplicate
        # (Camelot usually gives 100% accuracy, PDFPlumber is unrated so we default to 50)
        tables_sorted = sorted(
            tables,
            key=lambda t: t.get("accuracy", 0), 
            reverse=True
        )

        unique_tables = []
        seen_hashes = set()

        for table in tables_sorted:
            df = table["dataframe"]
            
            # 2. Create a Content Hash
            # We use the column names + the first value of the first row
            # This is robust: extraction might differ slightly, but headers usually match.
            if df.empty:
                continue
                
            # Serialize headers and first row to string for hashing
            headers = ",".join(map(str, df.columns.tolist()))
            first_row = ",".join(map(str, df.iloc[0].tolist())) if len(df) > 0 else ""
            
            # Unique Key: Page + Content
            content_hash = hash(f"{table['page']}_{headers}_{first_row}")

            if content_hash not in seen_hashes:
                unique_tables.append(table)
                seen_hashes.add(content_hash)
                
        return unique_tables

    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to markdown table format

        Args:
            df: Pandas DataFrame

        Returns:
            Markdown formatted string
        """
        try:
            # Clean up DataFrame
            df = df.fillna("")
            df = df.astype(str)

            # Convert to markdown
            markdown = df.to_markdown(index=False)

            return markdown

        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")

            # Fallback: manual conversion
            lines = []

            # Header
            headers = df.columns.tolist()
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")

            # Rows
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(v) for v in row) + " |")

            return "\n".join(lines)

    def table_to_text(self, table: Dict[str, Any], format: str = "markdown") -> str:
        """
        Convert table dictionary to text format

        Args:
            table: Table dictionary from extraction
            format: Output format ('markdown', 'csv', 'tsv')

        Returns:
            Formatted table string
        """
        df = table.get("dataframe")
        if df is None:
            return table.get("markdown", "")

        if format == "markdown":
            return table.get("markdown", "")

        elif format == "csv":
            return df.to_csv(index=False)

        elif format == "tsv":
            return df.to_csv(index=False, sep="\t")

        else:
            raise ValueError(f"Unknown format: {format}")
