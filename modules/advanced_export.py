"""
Advanced Export Module
Supports multiple export formats: Excel, JSON, CSV, SVG, PNG
"""

import io
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import plotly.io as pio

logger = logging.getLogger(__name__)


class AdvancedExporter:
    """Handles export of analysis results in multiple formats"""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
    
    def export_to_excel(self, gene_data, enrichment_results, selected_genes, 
                       analysis_params, integrity_score, session_metadata=None):
        """
        Export analysis to Excel with multiple sheets
        
        Args:
            gene_data: DataFrame with gene expression data
            enrichment_results: DataFrame or dict with enrichment analysis
            selected_genes: list of selected gene symbols
            analysis_params: dict with analysis parameters
            integrity_score: dict with data quality metrics
            session_metadata: dict with session information
        
        Returns:
            bytes - Excel file content
        """
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Selected Genes Detail
                if not gene_data.empty:
                    gene_col = next((c for c in ['gene_symbol', 'symbol'] if c in gene_data.columns), None)
                    selected_df = gene_data[gene_data[gene_col].isin(selected_genes)] if gene_col else pd.DataFrame()
                    if not selected_df.empty:
                        selected_df = selected_df.sort_values(
                            by='log2_fold_change' if 'log2_fold_change' in selected_df.columns else 'log2FC',
                            ascending=False
                        )
                        selected_df.to_excel(writer, sheet_name='Selected_Genes', index=False)
                
                # Sheet 2: All Gene Data (filtered by significance)
                if not gene_data.empty:
                    sig_df = gene_data.copy()
                    gene_data.to_excel(writer, sheet_name='All_Genes_Data', index=False)
                
                # Sheet 3: Pathway Enrichment Results
                if isinstance(enrichment_results, pd.DataFrame):
                    enrichment_results.to_excel(writer, sheet_name='Enrichment_Results', index=False)
                elif isinstance(enrichment_results, dict) and enrichment_results:
                    enrich_df = pd.DataFrame([enrichment_results])
                    enrich_df.to_excel(writer, sheet_name='Enrichment_Results', index=False)
                else:
                    # Empty sheet
                    pd.DataFrame({'Status': ['No enrichment results available']}).to_excel(
                        writer, sheet_name='Enrichment_Results', index=False
                    )
                
                # Sheet 4: Analysis Summary
                summary_data = {
                    'Metric': [
                        'Total Genes',
                        'Selected Genes',
                        'Log2FC Threshold',
                        'Adjusted P-value Threshold',
                        'Significant Genes (p<0.05)',
                        'Upregulated',
                        'Downregulated',
                        'Data Integrity Score',
                        'Integrity Grade',
                        'Export Timestamp'
                    ],
                    'Value': [
                        len(gene_data) if not gene_data.empty else 0,
                        len(selected_genes),
                        analysis_params.get('log2_fold_change', 'N/A'),
                        analysis_params.get('adjusted_p_value', 'N/A'),
                        integrity_score.get('n_significant', 0),
                        self._count_upregulated(gene_data, analysis_params),
                        self._count_downregulated(gene_data, analysis_params),
                        integrity_score.get('total', 'N/A'),
                        integrity_score.get('grade', 'N/A'),
                        self.timestamp
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 5: Analysis Parameters & Metadata
                params_data = {
                    'Parameter': list(analysis_params.keys()) + ['Export_Time'],
                    'Value': list(analysis_params.values()) + [self.timestamp]
                }
                if session_metadata:
                    for key, val in session_metadata.items():
                        params_data['Parameter'].append(key)
                        params_data['Value'].append(str(val))
                
                params_df = pd.DataFrame(params_data)
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            output.seek(0)
            return output.getvalue()
        
        except ImportError:
            logger.error("openpyxl not installed. Please install: pip install openpyxl")
            raise Exception("Excel export requires openpyxl library")
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise
    
    def export_to_json(self, gene_data, enrichment_results, selected_genes,
                      analysis_params, integrity_score, session_metadata=None):
        """
        Export complete analysis to JSON format
        
        Returns:
            bytes - JSON file content
        """
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': self.timestamp,
                    'tool_version': '3.0.0',
                    'session_info': session_metadata or {}
                },
                'analysis_parameters': analysis_params,
                'data_integrity': integrity_score,
                'gene_data': gene_data.to_dict('records') if not gene_data.empty else [],
                'selected_genes': selected_genes,
                'enrichment_results': enrichment_results.to_dict('records') 
                    if isinstance(enrichment_results, pd.DataFrame) 
                    else enrichment_results if isinstance(enrichment_results, dict) 
                    else {}
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            return json_str.encode('utf-8')
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def export_enrichment_csv(self, enrichment_results):
        """Export enrichment results to CSV"""
        try:
            if isinstance(enrichment_results, pd.DataFrame):
                return enrichment_results.to_csv(index=False).encode('utf-8')
            elif isinstance(enrichment_results, dict):
                df = pd.DataFrame([enrichment_results])
                return df.to_csv(index=False).encode('utf-8')
            else:
                return b'No enrichment results available'
        except Exception as e:
            logger.error(f"Error exporting enrichment CSV: {str(e)}")
            raise
    
    def export_selected_genes_csv(self, gene_data, selected_genes):
        """Export selected genes to CSV"""
        try:
            if gene_data.empty or not selected_genes:
                return b'No genes selected'
            
            gene_col = next((c for c in ['gene_symbol', 'symbol'] if c in gene_data.columns), None)
            if gene_col is None:
                return b'No gene symbol column found'
            selected_df = gene_data[gene_data[gene_col].isin(selected_genes)]
            return selected_df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            logger.error(f"Error exporting selected genes CSV: {str(e)}")
            raise
    
    def export_plot_image(self, figure, format='svg'):
        """
        Export Plotly figure to image format
        
        Args:
            figure: Plotly figure object
            format: 'svg', 'png', 'jpeg', 'webp'
        
        Returns:
            bytes - Image file content
        """
        try:
            supported_formats = ['svg', 'png', 'jpeg', 'webp']
            if format not in supported_formats:
                raise ValueError(f"Format must be one of {supported_formats}")
            
            # Export with high quality
            img_bytes = pio.to_image(figure, format=format, width=1200, height=800)
            return img_bytes
        except Exception as e:
            logger.error(f"Error exporting plot as {format}: {str(e)}")
            raise
    
    def _count_upregulated(self, gene_data, params):
        """Count upregulated genes based on thresholds"""
        try:
            if gene_data.empty:
                return 0
            
            lfc_col = 'log2_fold_change' if 'log2_fold_change' in gene_data.columns else 'log2FC'
            padj_col = 'adjusted_p_value' if 'adjusted_p_value' in gene_data.columns else 'padj'
            
            lfc_thresh = params.get('log2_fold_change', 1.0)
            padj_thresh = params.get('adjusted_p_value', 0.05)
            
            return int(((gene_data[lfc_col] > lfc_thresh) & 
                       (gene_data[padj_col] < padj_thresh)).sum())
        except:
            return 0
    
    def _count_downregulated(self, gene_data, params):
        """Count downregulated genes based on thresholds"""
        try:
            if gene_data.empty:
                return 0
            
            lfc_col = 'log2_fold_change' if 'log2_fold_change' in gene_data.columns else 'log2FC'
            padj_col = 'adjusted_p_value' if 'adjusted_p_value' in gene_data.columns else 'padj'
            
            lfc_thresh = params.get('log2_fold_change', 1.0)
            padj_thresh = params.get('adjusted_p_value', 0.05)
            
            return int(((gene_data[lfc_col] < -lfc_thresh) & 
                       (gene_data[padj_col] < padj_thresh)).sum())
        except:
            return 0


def get_exporter():
    """Get exporter instance"""
    return AdvancedExporter()
