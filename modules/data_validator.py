"""
Data Validation & Error Recovery Module
Provides comprehensive pre-flight validation with recovery suggestions
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error with recovery suggestions"""
    
    def __init__(self, level, category, message, recovery_suggestions=None):
        """
        Args:
            level: 'error', 'warning', 'info'
            category: type of issue (e.g., 'column_missing', 'data_format', 'outliers')
            message: detailed error message
            recovery_suggestions: list of suggested fixes
        """
        self.level = level
        self.category = category
        self.message = message
        self.recovery_suggestions = recovery_suggestions or []
    
    def to_dict(self):
        return {
            'level': self.level,
            'category': self.category,
            'message': self.message,
            'suggestions': self.recovery_suggestions
        }


class DataValidator:
    """Comprehensive data validation with recovery suggestions"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info_messages = []
    
    def validate_deg_file(self, df):
        """
        Comprehensive validation of DEG data with detailed error reporting
        
        Returns:
            tuple - (cleaned_df, validation_results)
        """
        self.errors = []
        self.warnings = []
        self.info_messages = []
        
        # Step 1: Check if DataFrame is empty
        if df.empty:
            self.errors.append(ValidationError(
                'error', 'empty_file',
                'The uploaded file is empty.',
                [
                    'Ensure the CSV file contains data rows (not just headers)',
                    'Check that the file was properly saved before upload',
                    'Try a different file format (ensure it\'s comma-separated)'
                ]
            ))
            return None
        
        # Step 2: Check columns
        self._validate_columns(df)
        if self.errors:  # Can't continue without proper columns
            return None
        
        # Rename columns to standard names for processing
        df_clean = self._standardize_columns(df)
        
        # Step 3: Check data types and convert
        self._validate_data_types(df_clean)
        if self.errors:
            return None
        
        # Step 4: Check for missing values
        self._validate_missing_values(df_clean)
        
        # Step 5: Check value ranges and outliers
        self._validate_value_ranges(df_clean)
        
        # Step 6: Check duplicates
        self._check_duplicates(df_clean)
        
        # Step 7: Check sample statistics
        self._validate_statistics(df_clean)
        
        # Step 8: Check gene symbol format
        self._validate_gene_symbols(df_clean)
        
        # Step 9: Clean the data
        df_clean = self._clean_data(df_clean)
        
        return df_clean
    
    def _validate_columns(self, df):
        """Check for required columns with flexible naming"""
        required_patterns = {
            'gene_symbol': ['symbol', 'gene', 'gene_name', 'id', 'name', 'geneid', 
                           'gene_id', 'genename', 'gene_symbol', 'hgnc'],
            'log2_fold_change': ['log2fc', 'logfc', 'lfc', 'log2foldchange', 
                                'log2_fold_change', 'fold_change', 'foldchange', 'log2_fc', 'fc'],
            'adjusted_p_value': ['padj', 'fdr', 'qvalue', 'p_adj', 'adj_p', 
                                'adjusted_p_value', 'adj.p.val', 'adj_pval', 'p.adj', 'pvalue']
        }
        
        col_map = {}
        missing_required = []
        
        for required, patterns in required_patterns.items():
            found = False
            for col in df.columns:
                col_lower = col.lower().replace(' ', '_').replace('-', '_')
                if col_lower in [p.lower().replace(' ', '_').replace('-', '_') for p in patterns]:
                    col_map[required] = col
                    found = True
                    break
            
            if not found:
                missing_required.append(required)
        
        if missing_required:
            suggestions = []
            if 'gene_symbol' in missing_required:
                suggestions.append("Rename gene identifier column to: 'symbol', 'gene_name', 'gene', or 'id'")
            if 'log2_fold_change' in missing_required:
                suggestions.append("Ensure fold change column is named one of: 'log2FC', 'LFC', 'log2_fold_change'")
            if 'adjusted_p_value' in missing_required:
                suggestions.append("Ensure p-value column is named one of: 'padj', 'FDR', 'qvalue', 'adjusted_p_value'")
            
            suggestions.append(f"Available columns in your file: {', '.join(df.columns.tolist())}")
            
            self.errors.append(ValidationError(
                'error', 'missing_columns',
                f"Missing required columns: {', '.join(missing_required). Expected at least one column for each: gene symbol, log2 fold change, adjusted p-value.",
                suggestions
            ))
        
        self.col_map = col_map
    
    def _standardize_columns(self, df):
        """Rename columns to standard names"""
        df = df.rename(columns=self.col_map)
        return df
    
    def _validate_data_types(self, df):
        """Validate and convert data types"""
        try:
            # Convert fold change to numeric
            df['log2_fold_change'] = pd.to_numeric(df['log2_fold_change'], errors='coerce')
            # Convert p-values to numeric
            df['adjusted_p_value'] = pd.to_numeric(df['adjusted_p_value'], errors='coerce')
            
            failed_fc = df['log2_fold_change'].isna().sum()
            failed_p = df['adjusted_p_value'].isna().sum()
            
            if failed_fc > 0 or failed_p > 0:
                self.warnings.append(ValidationError(
                    'warning', 'data_conversion',
                    f"Failed to convert {failed_fc} fold-change values and {failed_p} p-values to numbers (marked as invalid).",
                    [
                        'Check for text characters in numeric columns',
                        'Ensure decimal separator is correct (. not ,)',
                        'Remove currency symbols or other special characters from numbers'
                    ]
                ))
        
        except Exception as e:
            self.errors.append(ValidationError(
                'error', 'type_conversion',
                f'Data type conversion failed: {str(e)}',
                ['Verify column data types are correct', 'Check for unusual formatting']
            ))
    
    def _validate_missing_values(self, df):
        """Check for missing values"""
        original_len = len(df)
        missing_before = df[['gene_symbol', 'log2_fold_change', 'adjusted_p_value']].isna().sum()
        
        df_clean = df.dropna(subset=['gene_symbol', 'log2_fold_change', 'adjusted_p_value'])
        rows_removed = original_len - len(df_clean)
        
        if rows_removed > 0:
            pct = (rows_removed / original_len) * 100
            self.warnings.append(ValidationError(
                'warning', 'missing_values',
                f'Removed {rows_removed} rows ({pct:.1f}%) with missing required values.',
                [
                    'Review the original file for data entry errors',
                    f'Check if {rows_removed} rows are expected'  ,
                    'Consider using imputation if data is valuable'
                ]
            ))
        
        return df_clean
    
    def _validate_value_ranges(self, df):
        """Check for value ranges and outliers"""
        # Clip p-values to [0, 1]
        df['adjusted_p_value'] = df['adjusted_p_value'].clip(0, 1.0)
        
        # Check for negative p-values (shouldn't exist)
        if (df['adjusted_p_value'] < 0).any():
            self.errors.append(ValidationError(
                'error', 'invalid_pvalues',
                'Negative p-values detected (impossible for statistical tests).',
                ['Verify p-value column is correct', 'Check data source and processing']
            ))
        
        # Check for extremely large fold changes
        max_fc = df['log2_fold_change'].abs().max()
        if max_fc > 20:
            self.warnings.append(ValidationError(
                'warning', 'extreme_values',
                f'Extremely large fold changes detected (max: {max_fc:.2f}). This is unusual for RNA-Seq.',
                [
                    'Verify expression data was normalized correctly',
                    'Check if data is in log2 scale (not original scale)',
                    'Consider removing extreme outliers if they represent technical artifacts'
                ]
            ))
        
        # Check for distributions
        zero_fc = (df['log2_fold_change'] == 0).sum()
        if zero_fc / len(df) > 0.5:
            self.warnings.append(ValidationError(
                'warning', 'no_variation',
                f'{zero_fc} genes ({(zero_fc/len(df)*100):.1f}%) have zero fold change.',
                [
                    'This is unusual if you expected differential expression',
                    'Check if data represents control vs control comparison',
                    'Verify condition labels are correct in your analysis'
                ]
            ))
    
    def _check_duplicates(self, df):
        """Check for duplicate gene symbols"""
        dup_genes = df['gene_symbol'].duplicated().sum()
        if dup_genes > 0:
            pct = (dup_genes / len(df)) * 100
            self.warnings.append(ValidationError(
                'warning', 'duplicate_genes',
                f'Found {dup_genes} ({pct:.1f}%) duplicate gene symbols. Keeping first occurrence.',
                [
                    'Check if gene symbols should be unique',
                    'Consider aggregating by gene symbol (e.g., take max/min effect size)',
                    'Verify data source for unexpected duplicates'
                ]
            ))
            df = df.drop_duplicates(subset=['gene_symbol'], keep='first')
        
        return df
    
    def _validate_statistics(self, df):
        """Check distribution and statistical properties"""
        # Check p-value distribution
        sig_frac = (df['adjusted_p_value'] < 0.05).sum() / len(df)
        
        if sig_frac < 0.001:
            self.warnings.append(ValidationError(
                'warning', 'no_significant',
                f'Very few significant genes: {sig_frac*100:.2f}% with p<0.05.',
                [
                    'Try relaxing the p-value threshold to 0.01 or 0.1',
                    'Check if experiment has low effect sizes',
                    'Verify sample sizes were adequate'
                ]
            ))
        
        if sig_frac > 0.95:
            self.warnings.append(ValidationError(
                'warning', 'too_many_significant',
                f'Almost all genes significant: {sig_frac*100:.2f}% with p<0.05.',
                [
                    'This is unusual - may indicate batch effects',
                    'Try stricter p-value threshold (0.01)',
                    'Consider QC of your sequencing/array data'
                ]
            ))
        
        # Check fold-change distribution
        pos_frac = (df['log2_fold_change'] > 0).sum() / len(df)
        if abs(pos_frac - 0.5) > 0.3:
            self.warnings.append(ValidationError(
                'warning', 'asymmetric_fc',
                f'Asymmetric fold-change distribution: {pos_frac*100:.1f}% upregulated.',
                [
                    'Check if one condition has systematically higher expression',
                    'Verify normalization was performed correctly',
                    'Consider checking for batch effects'
                ]
            ))
    
    def _validate_gene_symbols(self, df):
        """Check gene symbol format"""
        # Check if they look like valid HGNC symbols
        valid_pattern = r'^[A-Z][A-Z0-9\-\.]{1,19}$'
        import re
        pattern = re.compile(valid_pattern)
        
        valid_count = df['gene_symbol'].astype(str).apply(lambda x: bool(pattern.match(x))).sum()
        valid_frac = valid_count / len(df)
        
        if valid_frac < 0.5:
            self.info_messages.append(ValidationError(
                'info', 'symbol_format',
                f'Only {valid_frac*100:.1f}% of gene symbols match HGNC format. May use alternative ID system.',
                [
                    'Ensembl IDs: ENSG00000...',
                    'Entrez IDs: numeric only',
                    'UniProt: P12345',
                    'Consider converting to HGNC symbols for better pathway annotation'
                ]
            ))
    
    def _clean_data(self, df):
        """Final data cleanup"""
        df = df.dropna(subset=['gene_symbol', 'log2_fold_change', 'adjusted_p_value'])
        df = df.drop_duplicates(subset=['gene_symbol'], keep='first')
        return df
    
    def get_results(self):
        """Get validation results summary"""
        return {
            'errors': [e.to_dict() for e in self.errors],
            'warnings': [w.to_dict() for w in self.warnings],
            'info': [i.to_dict() for i in self.info_messages],
            'has_errors': len(self.errors) > 0,
            'has_warnings': len(self.warnings) > 0
        }
    
    def get_summary_html(self):
        """Generate HTML summary of validation results"""
        html_parts = []
        
        # Errors section
        if self.errors:
            html_parts.append('<div class="alert alert-danger">')
            html_parts.append('<h6>❌ Critical Issues (must fix):</h6>')
            for error in self.errors:
                html_parts.append(f'<p><strong>{error.category}:</strong> {error.message}</p>')
                if error.recovery_suggestions:
                    html_parts.append('<ul style="margin: 4px 0; padding-left: 20px; font-size: 11px;">')
                    for suggestion in error.recovery_suggestions:
                        html_parts.append(f'<li>{suggestion}</li>')
                    html_parts.append('</ul>')
            html_parts.append('</div>')
        
        # Warnings section
        if self.warnings:
            html_parts.append('<div class="alert alert-warning">')
            html_parts.append('<h6>⚠️ Warnings (review recommended):</h6>')
            for warning in self.warnings:
                html_parts.append(f'<p><strong>{warning.category}:</strong> {warning.message}</p>')
                if warning.recovery_suggestions:
                    html_parts.append('<ul style="margin: 4px 0; padding-left: 20px; font-size: 11px;">')
                    for suggestion in warning.recovery_suggestions[:2]:  # Limit to 2 suggestions
                        html_parts.append(f'<li>{suggestion}</li>')
                    html_parts.append('</ul>')
            html_parts.append('</div>')
        
        # Info section
        if self.info_messages:
            html_parts.append('<div class="alert alert-info">')
            html_parts.append('<h6>ℹ️ Information:</h6>')
            for info in self.info_messages:
                html_parts.append(f'<p><small>{info.message}</small></p>')
            html_parts.append('</div>')
        
        return ''.join(html_parts)


def validate_deg_data(df):
    """
    Wrapper function for backward compatibility
    Returns: (clean_df, errors, warnings, html_summary)
    """
    validator = DataValidator()
    clean_df = validator.validate_deg_file(df)
    results = validator.get_results()
    
    errors = [e['message'] for e in results['errors']]
    warnings = [w['message'] for w in results['warnings']]
    html_summary = validator.get_summary_html()
    
    return clean_df, errors, warnings, html_summary
