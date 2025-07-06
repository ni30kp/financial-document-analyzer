"""
Index Management Utility for FAISS Vector Database
"""
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.settings import settings

logger = logging.getLogger(__name__)

class IndexManager:
    """
    Utility class for managing FAISS index operations
    """
    
    def __init__(self, index_path: str = None):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index
        
        Returns:
            Dictionary with index information
        """
        info = {
            "index_exists": False,
            "index_path": self.index_path,
            "file_sizes": {},
            "document_count": 0,
            "total_size_mb": 0.0
        }
        
        files = [
            f"{self.index_path}.index",
            f"{self.index_path}.metadata", 
            f"{self.index_path}.documents"
        ]
        
        total_size = 0
        for file_path in files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                info["file_sizes"][os.path.basename(file_path)] = size
                total_size += size
                info["index_exists"] = True
        
        info["total_size_mb"] = total_size / (1024 * 1024)
        
        # Try to get document count
        try:
            import pickle
            documents_file = f"{self.index_path}.documents"
            if os.path.exists(documents_file):
                with open(documents_file, 'rb') as f:
                    documents = pickle.load(f)
                    info["document_count"] = len(documents)
        except Exception as e:
            logger.warning(f"Could not read document count: {e}")
        
        return info
    
    def backup_index(self, backup_suffix: str = None) -> bool:
        """
        Create a backup of the current index
        
        Args:
            backup_suffix: Optional suffix for backup files
            
        Returns:
            True if backup successful, False otherwise
        """
        if not backup_suffix:
            from datetime import datetime
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            files = [
                f"{self.index_path}.index",
                f"{self.index_path}.metadata",
                f"{self.index_path}.documents"
            ]
            
            backup_count = 0
            for file_path in files:
                if os.path.exists(file_path):
                    backup_path = f"{file_path}.backup_{backup_suffix}"
                    shutil.copy2(file_path, backup_path)
                    backup_count += 1
            
            if backup_count > 0:
                logger.info(f"Created backup with suffix: {backup_suffix}")
                return True
            else:
                logger.warning("No index files found to backup")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def clear_index(self, create_backup: bool = True) -> bool:
        """
        Clear the current index
        
        Args:
            create_backup: Whether to create a backup before clearing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if create_backup:
                self.backup_index("before_clear")
            
            files = [
                f"{self.index_path}.index",
                f"{self.index_path}.metadata",
                f"{self.index_path}.documents"
            ]
            
            removed_count = 0
            for file_path in files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleared {removed_count} index files")
                return True
            else:
                logger.warning("No index files found to clear")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False
    
    def restore_backup(self, backup_suffix: str) -> bool:
        """
        Restore index from backup
        
        Args:
            backup_suffix: Suffix of backup files to restore
            
        Returns:
            True if successful, False otherwise
        """
        try:
            files = [
                f"{self.index_path}.index",
                f"{self.index_path}.metadata",
                f"{self.index_path}.documents"
            ]
            
            # First check if all backup files exist
            backup_files = [f"{f}.backup_{backup_suffix}" for f in files]
            for backup_file in backup_files:
                if not os.path.exists(backup_file):
                    logger.error(f"Backup file not found: {backup_file}")
                    return False
            
            # Restore files
            restored_count = 0
            for file_path, backup_file in zip(files, backup_files):
                shutil.copy2(backup_file, file_path)
                restored_count += 1
            
            logger.info(f"Restored {restored_count} files from backup: {backup_suffix}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """
        List available backups
        
        Returns:
            List of backup suffixes
        """
        backups = set()
        
        try:
            index_dir = os.path.dirname(self.index_path)
            if not os.path.exists(index_dir):
                return []
            
            for filename in os.listdir(index_dir):
                if filename.endswith('.backup_'):
                    continue
                if '.backup_' in filename:
                    suffix = filename.split('.backup_')[-1]
                    backups.add(suffix)
            
            return sorted(list(backups))
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def optimize_index(self) -> bool:
        """
        Optimize the index by removing duplicate documents
        
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            # This would require loading the index and removing duplicates
            # For now, we'll just log that optimization is needed
            logger.info("Index optimization not implemented yet")
            return False
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            return False
    
    def get_status_report(self) -> str:
        """
        Get a comprehensive status report
        
        Returns:
            Formatted status report string
        """
        info = self.get_index_info()
        backups = self.list_backups()
        
        report = f"""
Index Status Report
==================
Index Path: {info['index_path']}
Index Exists: {info['index_exists']}
Document Count: {info['document_count']}
Total Size: {info['total_size_mb']:.2f} MB

File Sizes:
{chr(10).join(f"  {name}: {size/1024:.1f} KB" for name, size in info['file_sizes'].items())}

Available Backups: {len(backups)}
{chr(10).join(f"  - {backup}" for backup in backups[:5])}
{"  ... and more" if len(backups) > 5 else ""}
"""
        return report.strip() 