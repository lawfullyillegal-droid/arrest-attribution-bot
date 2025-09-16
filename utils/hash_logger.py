"""
Hash Logger Module

This module provides secure logging and hashing utilities for sensitive data.
"""

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, List
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class HashLogEntry:
    """Hash log entry structure."""
    timestamp: datetime
    event_type: str
    data_hash: str
    metadata: Dict
    salt_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class HashLogger:
    """Secure logging utility that hashes sensitive data."""
    
    def __init__(self, log_file: Optional[str] = None, salt_key: Optional[str] = None):
        """
        Initialize the hash logger.
        
        Args:
            log_file: Path to log file (optional)
            salt_key: Salt key for hashing (optional, will generate if not provided)
        """
        self.log_file = log_file
        self.salt_key = salt_key or self._generate_salt_key()
        self.log_entries: List[HashLogEntry] = []
        
        # Setup file logging if path provided
        if self.log_file:
            self._setup_file_logging()
    
    def _generate_salt_key(self) -> str:
        """Generate a random salt key."""
        return os.urandom(32).hex()
    
    def _setup_file_logging(self):
        """Setup file logging configuration."""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
    
    def hash_data(self, data: Any, use_salt: bool = True) -> str:
        """
        Create a secure hash of the data.
        
        Args:
            data: Data to hash
            use_salt: Whether to use salt in hashing
            
        Returns:
            Hexadecimal hash string
        """
        # Convert data to string representation
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(data, default=str)
        else:
            data_str = str(data)
        
        # Create hash
        if use_salt:
            # Use HMAC for salted hash
            return hmac.new(
                self.salt_key.encode(),
                data_str.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            # Simple SHA256 hash
            return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_data_hash(self, data: Any, event_type: str, 
                     metadata: Optional[Dict] = None, use_salt: bool = True) -> str:
        """
        Log a hash of sensitive data.
        
        Args:
            data: Data to hash and log
            event_type: Type of event being logged
            metadata: Additional metadata to include
            use_salt: Whether to use salt in hashing
            
        Returns:
            Hash of the data
        """
        data_hash = self.hash_data(data, use_salt)
        
        # Create log entry
        entry = HashLogEntry(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            data_hash=data_hash,
            metadata=metadata or {},
            salt_id=self._get_salt_id() if use_salt else None
        )
        
        # Store entry
        self.log_entries.append(entry)
        
        # Log to file/console
        self._write_log_entry(entry)
        
        return data_hash
    
    def _get_salt_id(self) -> str:
        """Get identifier for the current salt (first 8 chars of salt hash)."""
        return hashlib.sha256(self.salt_key.encode()).hexdigest()[:8]
    
    def _write_log_entry(self, entry: HashLogEntry):
        """Write log entry to configured outputs."""
        log_message = f"HASH_LOG: {entry.event_type} | Hash: {entry.data_hash[:16]}... | Metadata: {entry.metadata}"
        
        if entry.salt_id:
            log_message += f" | Salt ID: {entry.salt_id}"
        
        logger.info(log_message)
    
    def verify_data_hash(self, data: Any, expected_hash: str, use_salt: bool = True) -> bool:
        """
        Verify that data produces the expected hash.
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            use_salt: Whether salt was used in original hash
            
        Returns:
            True if hash matches, False otherwise
        """
        computed_hash = self.hash_data(data, use_salt)
        return hmac.compare_digest(computed_hash, expected_hash)
    
    def log_pii_access(self, pii_type: str, record_id: str, 
                      access_reason: str, user_id: Optional[str] = None) -> str:
        """
        Log access to personally identifiable information.
        
        Args:
            pii_type: Type of PII accessed (e.g., 'name', 'address', 'ssn')
            record_id: Identifier of the record accessed
            access_reason: Reason for accessing the PII
            user_id: User who accessed the PII
            
        Returns:
            Hash of the access log
        """
        access_data = {
            'pii_type': pii_type,
            'record_id': record_id,
            'access_reason': access_reason,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return self.log_data_hash(
            access_data,
            'pii_access',
            metadata={
                'pii_type': pii_type,
                'user_id': user_id,
                'access_reason': access_reason
            }
        )
    
    def log_data_modification(self, original_data: Any, modified_data: Any, 
                            record_id: str, modification_type: str,
                            user_id: Optional[str] = None) -> Dict[str, str]:
        """
        Log data modification with before/after hashes.
        
        Args:
            original_data: Original data before modification
            modified_data: Data after modification
            record_id: Identifier of the modified record
            modification_type: Type of modification performed
            user_id: User who made the modification
            
        Returns:
            Dictionary with original and modified hashes
        """
        original_hash = self.hash_data(original_data)
        modified_hash = self.hash_data(modified_data)
        
        modification_log = {
            'record_id': record_id,
            'modification_type': modification_type,
            'original_hash': original_hash,
            'modified_hash': modified_hash,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.log_data_hash(
            modification_log,
            'data_modification',
            metadata={
                'record_id': record_id,
                'modification_type': modification_type,
                'user_id': user_id
            }
        )
        
        return {
            'original_hash': original_hash,
            'modified_hash': modified_hash
        }
    
    def create_audit_trail(self, record_id: str, actions: List[Dict]) -> str:
        """
        Create an audit trail for a record.
        
        Args:
            record_id: Identifier of the record
            actions: List of actions performed on the record
            
        Returns:
            Hash of the audit trail
        """
        audit_data = {
            'record_id': record_id,
            'actions': actions,
            'trail_created': datetime.now(timezone.utc).isoformat()
        }
        
        return self.log_data_hash(
            audit_data,
            'audit_trail',
            metadata={
                'record_id': record_id,
                'action_count': len(actions)
            }
        )
    
    def hash_sensitive_fields(self, data: Dict, sensitive_fields: List[str]) -> Dict:
        """
        Hash specific sensitive fields in a data dictionary.
        
        Args:
            data: Dictionary containing data
            sensitive_fields: List of field names to hash
            
        Returns:
            Dictionary with sensitive fields hashed
        """
        result = data.copy()
        
        for field in sensitive_fields:
            if field in result and result[field] is not None:
                original_value = result[field]
                hashed_value = self.hash_data(original_value)
                result[field] = f"HASHED:{hashed_value[:16]}..."
                
                # Log the field hashing
                self.log_data_hash(
                    original_value,
                    'field_hash',
                    metadata={
                        'field_name': field,
                        'original_type': type(original_value).__name__
                    }
                )
        
        return result
    
    def export_logs(self, output_file: Optional[str] = None) -> List[Dict]:
        """
        Export all log entries.
        
        Args:
            output_file: Optional file to write logs to
            
        Returns:
            List of log entry dictionaries
        """
        log_data = [entry.to_dict() for entry in self.log_entries]
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(log_data)} log entries to {output_file}")
        
        return log_data
    
    def get_log_stats(self) -> Dict:
        """Get statistics about logged entries."""
        if not self.log_entries:
            return {}
        
        event_types = {}
        for entry in self.log_entries:
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
        
        return {
            'total_entries': len(self.log_entries),
            'event_types': event_types,
            'first_entry': self.log_entries[0].timestamp.isoformat() if self.log_entries else None,
            'last_entry': self.log_entries[-1].timestamp.isoformat() if self.log_entries else None,
            'salt_id': self._get_salt_id()
        }
    
    def clear_logs(self, confirm: bool = False):
        """
        Clear all log entries.
        
        Args:
            confirm: Must be True to actually clear logs
        """
        if confirm:
            self.log_entries.clear()
            logger.warning("Hash log entries cleared")
        else:
            logger.warning("Log clear requested but not confirmed")


def create_secure_id(data: Any, prefix: str = "") -> str:
    """
    Create a secure, deterministic ID from data.
    
    Args:
        data: Data to create ID from
        prefix: Optional prefix for the ID
        
    Returns:
        Secure ID string
    """
    hasher = hashlib.sha256()
    
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    hasher.update(data_str.encode())
    hash_hex = hasher.hexdigest()[:12]  # Use first 12 characters
    
    if prefix:
        return f"{prefix}_{hash_hex}"
    return hash_hex


def main():
    """Main function for testing the hash logger."""
    # Create hash logger
    hash_logger = HashLogger()
    
    # Test data hashing
    test_data = {
        'name': 'John Doe',
        'ssn': '123-45-6789',
        'address': '123 Main St'
    }
    
    print("Testing hash logger:")
    
    # Hash the entire record
    record_hash = hash_logger.log_data_hash(test_data, 'arrest_record')
    print(f"Record hash: {record_hash[:16]}...")
    
    # Log PII access
    pii_hash = hash_logger.log_pii_access('name', 'record_001', 'investigation', 'officer_123')
    print(f"PII access hash: {pii_hash[:16]}...")
    
    # Hash sensitive fields
    hashed_data = hash_logger.hash_sensitive_fields(test_data, ['ssn', 'name'])
    print(f"Data with hashed fields: {hashed_data}")
    
    # Create secure ID
    secure_id = create_secure_id(test_data, 'arrest')
    print(f"Secure ID: {secure_id}")
    
    # Print stats
    stats = hash_logger.get_log_stats()
    print(f"Log statistics: {stats}")


if __name__ == "__main__":
    main()