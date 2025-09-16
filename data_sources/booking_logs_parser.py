"""
Booking Logs Parser Module

This module handles parsing and processing of booking log files.
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class BookingLogsParser:
    """Parses booking logs from various file formats."""
    
    def __init__(self):
        """Initialize the parser."""
        self.supported_formats = ['.json', '.csv', '.xml', '.txt']
    
    def parse_file(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Parse a booking log file based on its format.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of parsed booking records
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return self._parse_json(file_path)
        elif extension == '.csv':
            return self._parse_csv(file_path)
        elif extension == '.xml':
            return self._parse_xml(file_path)
        elif extension == '.txt':
            return self._parse_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {extension}")
            return []
    
    def _parse_json(self, file_path: Path) -> List[Dict]:
        """Parse JSON booking logs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                logger.error(f"Unexpected JSON structure in {file_path}")
                return []
            
            # Normalize the records
            normalized_records = [self._normalize_record(record) for record in records]
            
            logger.info(f"Parsed {len(normalized_records)} records from JSON file: {file_path}")
            return normalized_records
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def _parse_csv(self, file_path: Path) -> List[Dict]:
        """Parse CSV booking logs."""
        try:
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                
                delimiter = ',' if ',' in sample else '\t'
                
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    normalized_record = self._normalize_record(row)
                    records.append(normalized_record)
            
            logger.info(f"Parsed {len(records)} records from CSV file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to parse CSV file {file_path}: {e}")
            return []
    
    def _parse_xml(self, file_path: Path) -> List[Dict]:
        """Parse XML booking logs."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            records = []
            
            # Look for common XML structures
            for booking in root.findall('.//booking') or root.findall('.//record') or [root]:
                record = {}
                for child in booking:
                    record[child.tag] = child.text
                
                if record:  # Only add non-empty records
                    normalized_record = self._normalize_record(record)
                    records.append(normalized_record)
            
            logger.info(f"Parsed {len(records)} records from XML file: {file_path}")
            return records
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading XML file {file_path}: {e}")
            return []
    
    def _parse_text(self, file_path: Path) -> List[Dict]:
        """Parse plain text booking logs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            records = []
            
            # Basic pattern matching for common text log formats
            # This would need to be customized based on actual log formats
            patterns = {
                'booking_number': r'Booking[#\s]*:?\s*([A-Z0-9\-]+)',
                'name': r'Name[:\s]*([A-Za-z\s,]+)',
                'charge': r'Charge[:\s]*([A-Za-z0-9\s\-,\.]+)',
                'date': r'Date[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            }
            
            # Split content into potential records
            # This is a simple implementation that would need refinement
            sections = re.split(r'\n\s*\n', content)
            
            for section in sections:
                record = {}
                for field, pattern in patterns.items():
                    match = re.search(pattern, section, re.IGNORECASE)
                    if match:
                        record[field] = match.group(1).strip()
                
                if record:  # Only add records with at least some data
                    normalized_record = self._normalize_record(record)
                    records.append(normalized_record)
            
            logger.info(f"Parsed {len(records)} records from text file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {e}")
            return []
    
    def _normalize_record(self, record: Dict) -> Dict:
        """
        Normalize a booking record to a standard format.
        
        Args:
            record: Raw record data
            
        Returns:
            Normalized record
        """
        normalized = {
            'booking_number': None,
            'name': None,
            'charge': None,
            'booking_date': None,
            'officer': None,
            'department': None,
            'raw_data': record.copy()
        }
        
        # Map common field variations to standard fields
        field_mappings = {
            'booking_number': ['booking_number', 'booking_id', 'book_num', 'id'],
            'name': ['name', 'defendant_name', 'arrestee_name', 'full_name'],
            'charge': ['charge', 'charges', 'offense', 'violation'],
            'booking_date': ['booking_date', 'arrest_date', 'date', 'timestamp'],
            'officer': ['officer', 'arresting_officer', 'officer_name'],
            'department': ['department', 'agency', 'dept', 'jurisdiction']
        }
        
        for normalized_field, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in record and record[field]:
                    normalized[normalized_field] = str(record[field]).strip()
                    break
        
        return normalized
    
    def parse_directory(self, directory_path: Union[str, Path]) -> List[Dict]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing booking log files
            
        Returns:
            Combined list of all parsed records
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        all_records = []
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                records = self.parse_file(file_path)
                all_records.extend(records)
        
        logger.info(f"Parsed {len(all_records)} total records from directory: {directory_path}")
        return all_records


def main():
    """Main function for testing the parser."""
    parser = BookingLogsParser()
    print("Booking Logs Parser initialized")
    print(f"Supported formats: {parser.supported_formats}")


if __name__ == "__main__":
    main()