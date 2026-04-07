"""
Gene Dashboard - Industry-Grade RNA-Seq Analysis Platform
========================================================

This module provides data input/output utilities for the Gene Dashboard application.

Features:
- CSV/TSV file parsing with validation
- Metadata handling for multi-sample experiments
- File size and content validation
- Error handling with user-friendly messages

Author: Gene Dashboard Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import io
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_GENES = 100
MAX_GENES = 50000
MIN_SAMPLES = 2
MAX_SAMPLES = 1000

# Required columns for different input types
COUNT_MATRIX_REQUIRED = ['gene_symbol']
METADATA_REQUIRED = ['sample_id', 'condition']
DEG_REQUIRED = ['gene_symbol', 'log2_fold_change', 'adjusted_p_value']


class DataValidator:
    """Validates input data files and provides detailed error reporting."""

    @staticmethod
    def validate_file_size(content: bytes, filename: str) -> Tuple[bool, str]:
        """Check file size against limits."""
        size = len(content)
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {size/1024/1024:.1f}MB (max {MAX_FILE_SIZE/1024/1024:.0f}MB)"
        return True, ""

    @staticmethod
    def detect_delimiter(content: str) -> str:
        """Auto-detect CSV delimiter."""
        sample = content[:1024]
        commas = sample.count(',')
        tabs = sample.count('\t')
        semicolons = sample.count(';')

        if tabs > commas and tabs > semicolons:
            return '\t'
        elif commas > semicolons:
            return ','
        else:
            return ';'

    @staticmethod
    def validate_count_matrix(df: pd.DataFrame) -> Tuple[bool, list, list]:
        """Validate count matrix format."""
        errors = []
        warnings = []

        # Check required columns
        if 'gene_symbol' not in df.columns:
            # Try common alternatives
            for alt in ['symbol', 'gene', 'gene_name', 'id', 'name']:
                if alt in df.columns:
                    df.rename(columns={alt: 'gene_symbol'}, inplace=True)
                    break
            else:
                errors.append("Missing gene_symbol column")

        # Check for numeric count columns (samples)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sample_cols = [col for col in numeric_cols if col != 'gene_symbol']

        if len(sample_cols) < MIN_SAMPLES:
            errors.append(f"Too few sample columns: {len(sample_cols)} (minimum {MIN_SAMPLES})")
        elif len(sample_cols) > MAX_SAMPLES:
            warnings.append(f"Many sample columns: {len(sample_cols)} (maximum {MAX_SAMPLES})")

        # Check gene count
        if len(df) < MIN_GENES:
            errors.append(f"Too few genes: {len(df)} (minimum {MIN_GENES})")
        elif len(df) > MAX_GENES:
            warnings.append(f"Many genes: {len(df)} (maximum {MAX_GENES})")

        # Check for missing values
        missing_pct = df[sample_cols].isnull().sum().sum() / (len(df) * len(sample_cols))
        if missing_pct > 0.1:
            errors.append(f"Too many missing values: {missing_pct:.1%}")

        # Check for negative values
        negative_count = (df[sample_cols] < 0).sum().sum()
        if negative_count > 0:
            errors.append(f"Negative count values found: {negative_count} instances")

        return len(errors) == 0, errors, warnings

    @staticmethod
    def validate_metadata(df: pd.DataFrame) -> Tuple[bool, list, list]:
        """Validate metadata format."""
        errors = []
        warnings = []

        # Check required columns
        missing_cols = []
        for col in METADATA_REQUIRED:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        # Check sample count
        if len(df) < MIN_SAMPLES:
            errors.append(f"Too few samples: {len(df)} (minimum {MIN_SAMPLES})")
        elif len(df) > MAX_SAMPLES:
            warnings.append(f"Many samples: {len(df)} (maximum {MAX_SAMPLES})")

        # Check for duplicate sample IDs
        if df['sample_id'].duplicated().any():
            errors.append("Duplicate sample IDs found")

        # Check condition labels
        conditions = df['condition'].value_counts()
        if len(conditions) < 2:
            errors.append("At least 2 different conditions required")

        return len(errors) == 0, errors, warnings

    @staticmethod
    def validate_deg_data(df: pd.DataFrame) -> Tuple[bool, list, list]:
        """Validate DEG data format (existing functionality)."""
        errors = []
        warnings = []

        # Check required columns
        missing_cols = []
        for col in DEG_REQUIRED:
            if col not in df.columns:
                # Try common alternatives
                alternatives = {
                    'gene_symbol': ['symbol', 'gene', 'gene_name', 'id', 'name'],
                    'log2_fold_change': ['log2FC', 'logfc', 'lfc', 'fold_change'],
                    'adjusted_p_value': ['padj', 'fdr', 'qvalue', 'adj_p']
                }
                found_alt = None
                for alt in alternatives.get(col, []):
                    if alt in df.columns:
                        found_alt = alt
                        break
                if found_alt:
                    df.rename(columns={found_alt: col}, inplace=True)
                else:
                    missing_cols.append(col)

        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        if errors:
            return False, errors, warnings

        # Data type validation
        try:
            df['log2_fold_change'] = pd.to_numeric(df['log2_fold_change'], errors='coerce')
            df['adjusted_p_value'] = pd.to_numeric(df['adjusted_p_value'], errors='coerce')
        except Exception as e:
            errors.append(f"Data type conversion failed: {str(e)}")

        # Remove rows with missing values
        original_len = len(df)
        df = df.dropna(subset=['gene_symbol', 'log2_fold_change', 'adjusted_p_value'])
        if len(df) < original_len:
            warnings.append(f"Removed {original_len - len(df)} rows with missing values")

        # Validate data ranges
        if (df['adjusted_p_value'] < 0).any() or (df['adjusted_p_value'] > 1).any():
            errors.append("Adjusted p-values must be between 0 and 1")

        if df['log2_fold_change'].abs().max() > 20:
            warnings.append("Extremely large fold changes detected (>20). Please verify data scaling.")

        # Check for duplicates
        dup_genes = df['gene_symbol'].duplicated().sum()
        if dup_genes > 0:
            warnings.append(f"Found {dup_genes} duplicate gene symbols. Keeping first occurrence.")
            df = df.drop_duplicates(subset=['gene_symbol'], keep='first')

        # Check data size
        if len(df) < MIN_GENES:
            errors.append(f"Dataset too small: {len(df)} genes. Minimum required: {MIN_GENES}")

        if len(df) > MAX_GENES:
            warnings.append(f"Large dataset: {len(df)} genes. Performance may be affected.")

        # Statistical quality checks
        n_sig = ((df['log2_fold_change'].abs() > 1) & (df['adjusted_p_value'] < 0.05)).sum()
        if n_sig < 10:
            warnings.append("Very few significantly differentially expressed genes detected. Consider relaxing thresholds.")

        return len(errors) == 0, errors, warnings


class DataLoader:
    """Handles loading and parsing of various data file formats."""

    @staticmethod
    def load_count_matrix(content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], list, list]:
        """Load and validate count matrix from uploaded file."""
        # Validate file size
        valid_size, size_msg = DataValidator.validate_file_size(content, filename)
        if not valid_size:
            return None, [size_msg], []

        try:
            # Decode content
            text = content.decode('utf-8')

            # Detect delimiter
            delimiter = DataValidator.detect_delimiter(text)

            # Parse CSV
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter)

            # Validate format
            valid, errors, warnings = DataValidator.validate_count_matrix(df)

            if not valid:
                return None, errors, warnings

            return df, errors, warnings

        except Exception as e:
            logger.error(f"Error loading count matrix: {e}")
            return None, [f"Failed to parse file: {str(e)}"], []

    @staticmethod
    def load_metadata(content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], list, list]:
        """Load and validate metadata from uploaded file."""
        # Validate file size
        valid_size, size_msg = DataValidator.validate_file_size(content, filename)
        if not valid_size:
            return None, [size_msg], []

        try:
            # Decode content
            text = content.decode('utf-8')

            # Detect delimiter
            delimiter = DataValidator.detect_delimiter(text)

            # Parse CSV
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter)

            # Validate format
            valid, errors, warnings = DataValidator.validate_metadata(df)

            if not valid:
                return None, errors, warnings

            return df, errors, warnings

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None, [f"Failed to parse file: {str(e)}"], []

    @staticmethod
    def load_deg_data(content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], list, list]:
        """Load and validate DEG data from uploaded file (existing functionality)."""
        # Validate file size
        valid_size, size_msg = DataValidator.validate_file_size(content, filename)
        if not valid_size:
            return None, [size_msg], []

        try:
            # Decode content
            text = content.decode('utf-8')

            # Parse CSV
            df = pd.read_csv(io.StringIO(text))

            # Validate format
            valid, errors, warnings = DataValidator.validate_deg_data(df)

            if not valid:
                return None, errors, warnings

            return df, errors, warnings

        except Exception as e:
            logger.error(f"Error loading DEG data: {e}")
            return None, [f"Failed to parse file: {str(e)}"], []


class DataExporter:
    """Handles data export functionality."""

    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str = "export.csv") -> Dict[str, Any]:
        """Export DataFrame to CSV format for download."""
        csv_string = df.to_csv(index=False)
        return {
            'content': csv_string,
            'filename': filename,
            'type': 'text/csv'
        }

    @staticmethod
    def export_to_json(data: Dict[str, Any], filename: str = "export.json") -> Dict[str, Any]:
        """Export data dictionary to JSON format for download."""
        import json
        json_string = json.dumps(data, indent=2, default=str)
        return {
            'content': json_string,
            'filename': filename,
            'type': 'application/json'
        }