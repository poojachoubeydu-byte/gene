"""
Batch Analysis Module
Handles processing of multiple DEG files with comparison functionality
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchAnalysisJob:
    """Represents a single batch analysis job"""
    
    def __init__(self, job_id, files_metadata):
        self.job_id = job_id
        self.files_metadata = files_metadata  # List of {filename, filedata}
        self.results = {}
        self.created_at = datetime.now().isoformat()
        self.status = 'pending'  # pending, processing, completed, failed
        self.error_message = None
    
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'status': self.status,
            'created_at': self.created_at,
            'files_count': len(self.files_metadata),
            'completed_files': len([r for r in self.results.values() if r.get('status') == 'completed']),
            'error_message': self.error_message
        }


class BatchAnalyzer:
    """Process multiple gene expression files in batch mode"""
    
    def __init__(self):
        self.jobs = {}
    
    def create_batch_job(self, job_id, files_metadata):
        """Create a new batch analysis job"""
        job = BatchAnalysisJob(job_id, files_metadata)
        self.jobs[job_id] = job
        return job
    
    def process_files(self, job_id, validator_func, params=None):
        """
        Process all files in a batch job
        
        Args:
            job_id: Job identifier
            validator_func: Function to validate and process each file
            params: Analysis parameters
        
        Returns:
            dict with processing results
        """
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        job = self.jobs[job_id]
        job.status = 'processing'
        params = params or {}
        
        for i, file_meta in enumerate(job.files_metadata):
            try:
                filename = file_meta.get('filename', f'file_{i}')
                df = file_meta.get('data')
                
                # Validate and process
                clean_df, errors, warnings, html_summary = validator_func(df)
                
                if errors:
                    job.results[filename] = {
                        'status': 'failed',
                        'error': '; '.join(errors),
                        'warnings': warnings
                    }
                else:
                    # Process successful
                    summary = self._compute_file_summary(clean_df, filename, params)
                    job.results[filename] = {
                        'status': 'completed',
                        'summary': summary,
                        'data': clean_df.to_dict('records'),
                        'warnings': warnings
                    }
            
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                job.results[filename] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        job.status = 'completed'
        return job.to_dict()
    
    def _compute_file_summary(self, df, filename, params):
        """Compute analysis summary for a file"""
        try:
            lfc_col = 'log2_fold_change' if 'log2_fold_change' in df.columns else 'log2FC'
            padj_col = 'adjusted_p_value' if 'adjusted_p_value' in df.columns else 'padj'
            lfc_thresh = params.get('log2_fold_change', 1.0)
            padj_thresh = params.get('adjusted_p_value', 0.05)
            
            sig = (df[padj_col] < padj_thresh)
            up = sig & (df[lfc_col] > lfc_thresh)
            down = sig & (df[lfc_col] < -lfc_thresh)
            
            return {
                'filename': filename,
                'total_genes': len(df),
                'significant_genes': int(sig.sum()),
                'upregulated': int(up.sum()),
                'downregulated': int(down.sum()),
                'mean_fc': float(df[lfc_col].mean()),
                'max_fc': float(df[lfc_col].abs().max()),
                'min_pvalue': float(df[padj_col].min()) if not df.empty else 1.0
            }
        except Exception as e:
            logger.error(f"Summary computation error: {str(e)}")
            return {'error': str(e)}
    
    def get_comparison_analysis(self, job_id):
        """Generate comparative analysis across batch samples"""
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        job = self.jobs[job_id]
        
        if not job.results:
            return {'error': 'No processed results available'}
        
        summaries = []
        all_genes = set()
        
        for filename, result in job.results.items():
            if result.get('status') == 'completed':
                summaries.append(result['summary'])
                if 'data' in result:
                    df = pd.DataFrame(result['data'])
                    gene_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'symbol'
                    if gene_col in df.columns:
                        all_genes.update(df[gene_col].tolist())
        
        if not summaries:
            return {'error': 'No completed analyses for comparison'}
        
        # Build comparison table
        comparison_df = pd.DataFrame(summaries)
        
        # Calculate overlaps
        overlap_analysis = self._calculate_overlaps(job.results)
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'overlap_analysis': overlap_analysis,
            'total_unique_genes': len(all_genes),
            'samples_count': len(summaries)
        }
    
    def _calculate_overlaps(self, results):
        """Calculate gene overlaps between samples"""
        try:
            sample_genes = {}
            
            for filename, result in results.items():
                if result.get('status') == 'completed' and 'data' in result:
                    df = pd.DataFrame(result['data'])
                    gene_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'symbol'
                    lfc_col = 'log2_fold_change' if 'log2_fold_change' in df.columns else 'log2FC'
                    padj_col = 'adjusted_p_value' if 'adjusted_p_value' in df.columns else 'padj'
                    
                    if gene_col in df.columns:
                        # Get significant genes
                        sig = (df[padj_col] < 0.05) & (df[lfc_col].abs() > 1.0)
                        sample_genes[filename] = set(df[sig][gene_col].tolist())
            
            if len(sample_genes) < 2:
                return {}
            
            # Calculate pairwise overlaps
            overlaps = {}
            filenames = list(sample_genes.keys())
            
            for i, fname1 in enumerate(filenames):
                for fname2 in filenames[i+1:]:
                    overlap = sample_genes[fname1] & sample_genes[fname2]
                    union = sample_genes[fname1] | sample_genes[fname2]
                    
                    key = f"{fname1} ∩ {fname2}"
                    overlaps[key] = {
                        'intersection': len(overlap),
                        'union': len(union),
                        'jaccard': len(overlap) / len(union) if union else 0,
                        'genes': list(overlap)[:10]  # Top 10 for preview
                    }
            
            return overlaps
        
        except Exception as e:
            logger.error(f"Overlap calculation error: {str(e)}")
            return {}
    
    def export_batch_results(self, job_id, format='xlsx'):
        """Export batch analysis results"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        try:
            if format == 'xlsx':
                import io
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Summary comparison
                    summaries = [r['summary'] for r in job.results.values() 
                               if r.get('status') == 'completed' and 'summary' in r]
                    if summaries:
                        pd.DataFrame(summaries).to_excel(writer, sheet_name='Comparison', index=False)
                    
                    # Sheet 2-N: Individual file results
                    for i, (filename, result) in enumerate(job.results.items()):
                        if result.get('status') == 'completed' and 'data' in result:
                            df = pd.DataFrame(result['data'])
                            sheet_name = filename.replace('.csv', '')[:31]  # Excel limit
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                output.seek(0)
                return output.getvalue()
        
        except ImportError:
            logger.error("openpyxl not available for export")
            return None
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            return None
    
    def get_job_status(self, job_id):
        """Get status of a batch job"""
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id].to_dict()
    
    def list_jobs(self, limit=10):
        """List recent batch jobs"""
        return sorted(
            [job.to_dict() for job in self.jobs.values()],
            key=lambda x: x['created_at'],
            reverse=True
        )[:limit]
    
    def cleanup_job(self, job_id):
        """Remove a completed batch job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False


# Global batch analyzer singleton
_batch_analyzer = None


def get_batch_analyzer():
    """Get or create batch analyzer instance"""
    global _batch_analyzer
    if _batch_analyzer is None:
        _batch_analyzer = BatchAnalyzer()
    return _batch_analyzer
