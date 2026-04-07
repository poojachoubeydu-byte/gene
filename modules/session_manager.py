"""
Session Management Module
Handles save/load of complete analysis sessions including data, parameters, and results
"""

import diskcache
import json
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Session cache directory
SESSIONS_DIR = Path('./sessions_cache')
SESSIONS_DIR.mkdir(exist_ok=True)

class SessionManager:
    """Manages persistent analysis sessions"""
    
    def __init__(self, cache_dir='./sessions_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir))
    
    def save_session(self, session_name, session_data):
        """
        Save a complete analysis session
        
        Args:
            session_name: str - Unique session identifier
            session_data: dict - Contains:
                - 'data': DataFrame with gene expression
                - 'parameters': dict with LFC, p-value thresholds
                - 'enrichment_results': dict with pathway analysis
                - 'selected_genes': list
                - 'metadata': dict with timestamp, description
        
        Returns:
            dict - Status with success and message
        """
        try:
            # Sanitize session name
            session_name = str(session_name).replace(' ', '_').replace('/', '_')[:50]
            
            # Add timestamp to metadata if not present
            if 'metadata' not in session_data:
                session_data['metadata'] = {}
            session_data['metadata']['saved_at'] = datetime.now().isoformat()
            session_data['metadata']['session_name'] = session_name
            
            # Convert DataFrame to serializable format
            serializable_data = self._serialize_session(session_data)
            
            # Store in cache
            key = f"session:{session_name}"
            self.cache[key] = serializable_data

            logger.info(f"Session saved: {session_name}")
            return {
                'success': True,
                'message': f'Session "{session_name}" saved successfully',
                'session_name': session_name,
                'timestamp': session_data['metadata']['saved_at']
            }
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            return {
                'success': False,
                'message': f'Error saving session: {str(e)}'
            }
    
    def load_session(self, session_name):
        """
        Load a previously saved analysis session
        
        Args:
            session_name: str - Session identifier to load
        
        Returns:
            tuple - (session_data dict, status dict)
        """
        try:
            session_name = str(session_name).replace(' ', '_').replace('/', '_')[:50]
            key = f"session:{session_name}"
            
            # Retrieve from cache
            if key in self.cache:
                serializable_data = self.cache[key]
                session_data = self._deserialize_session(serializable_data)
                
                logger.info(f"Session loaded: {session_name}")
                return session_data, {
                    'success': True,
                    'message': f'Session "{session_name}" loaded successfully'
                }
            else:
                return None, {
                    'success': False,
                    'message': f'Session "{session_name}" not found'
                }
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            return None, {
                'success': False,
                'message': f'Error loading session: {str(e)}'
            }
    
    def list_sessions(self):
        """
        List all available sessions
        
        Returns:
            list - List of session info dicts with name, timestamp, gene_count
        """
        try:
            sessions = []
            for key in self.cache.iterkeys():
                if key.startswith('session:'):
                    session_name = key.replace('session:', '')
                    data = self.cache[key]
                    
                    sessions.append({
                        'name': session_name,
                        'timestamp': data.get('metadata', {}).get('saved_at', 'Unknown'),
                        'gene_count': len(data.get('selected_genes', [])),
                        'description': data.get('metadata', {}).get('description', '')
                    })
            
            # Sort by timestamp, newest first
            sessions.sort(key=lambda x: x['timestamp'], reverse=True)
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []
    
    def delete_session(self, session_name):
        """
        Delete a saved session
        
        Args:
            session_name: str
        
        Returns:
            dict - Status
        """
        try:
            session_name = str(session_name).replace(' ', '_').replace('/', '_')[:50]
            key = f"session:{session_name}"
            
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Session deleted: {session_name}")
                return {
                    'success': True,
                    'message': f'Session "{session_name}" deleted'
                }
            else:
                return {
                    'success': False,
                    'message': f'Session "{session_name}" not found'
                }
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return {
                'success': False,
                'message': f'Error deleting session: {str(e)}'
            }
    
    def _serialize_session(self, session_data):
        """Convert session data to JSON-serializable format"""
        serializable = {}
        
        # Handle DataFrame
        if 'data' in session_data and isinstance(session_data['data'], pd.DataFrame):
            serializable['data'] = session_data['data'].to_dict('records')
        else:
            serializable['data'] = session_data.get('data', [])
        
        # Handle parameters (should already be dict)
        serializable['parameters'] = session_data.get('parameters', {})
        
        # Handle enrichment results
        if 'enrichment_results' in session_data:
            enrichment = session_data['enrichment_results']
            if isinstance(enrichment, pd.DataFrame):
                serializable['enrichment_results'] = enrichment.to_dict('records')
            else:
                serializable['enrichment_results'] = enrichment
        else:
            serializable['enrichment_results'] = {}
        
        # Handle selected genes
        serializable['selected_genes'] = list(session_data.get('selected_genes', []))
        
        # Handle metadata
        serializable['metadata'] = session_data.get('metadata', {})
        
        return serializable
    
    def _deserialize_session(self, serializable_data):
        """Convert JSON-serialized data back to session format"""
        session_data = {}
        
        # Reconstruct DataFrame
        if serializable_data.get('data'):
            session_data['data'] = pd.DataFrame(serializable_data['data'])
        else:
            session_data['data'] = pd.DataFrame()
        
        # Parameters
        session_data['parameters'] = serializable_data.get('parameters', {})
        
        # Enrichment results
        enrichment_raw = serializable_data.get('enrichment_results', {})
        if isinstance(enrichment_raw, list) and enrichment_raw:
            session_data['enrichment_results'] = pd.DataFrame(enrichment_raw)
        else:
            session_data['enrichment_results'] = enrichment_raw if isinstance(enrichment_raw, dict) else {}
        
        # Selected genes
        session_data['selected_genes'] = serializable_data.get('selected_genes', [])
        
        # Metadata
        session_data['metadata'] = serializable_data.get('metadata', {})
        
        return session_data
    
    def export_session_summary(self, session_data):
        """
        Generate a summary of the session for display/export
        
        Returns:
            dict - Summary with key statistics
        """
        try:
            data = session_data.get('data', pd.DataFrame())
            params = session_data.get('parameters', {})
            selected_genes = session_data.get('selected_genes', [])
            metadata = session_data.get('metadata', {})
            
            summary = {
                'session_info': {
                    'name': metadata.get('session_name', 'Unnamed'),
                    'description': metadata.get('description', ''),
                    'created': metadata.get('saved_at', ''),
                },
                'data_summary': {
                    'total_genes': len(data),
                    'selected_genes': len(selected_genes),
                    'columns': list(data.columns) if not data.empty else []
                },
                'analysis_parameters': {
                    'lfc_threshold': params.get('log2_fold_change', 'N/A'),
                    'p_value_threshold': params.get('adjusted_p_value', 'N/A')
                },
                'enrichment_count': len(session_data.get('enrichment_results', {}))
            }
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}


# Global session manager instance
_session_manager = None

def get_session_manager():
    """Get or create session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
