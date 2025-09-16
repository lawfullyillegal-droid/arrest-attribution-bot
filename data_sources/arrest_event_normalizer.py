"""
Arrest Event Normalizer Module

This module normalizes arrest event data from various sources into a standardized format.
"""

from typing import List, Dict, Optional, Union
from datetime import datetime, date, timezone
import re
import logging
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class NormalizedArrestEvent:
    """Standardized arrest event data structure."""
    event_id: str
    booking_number: Optional[str]
    arrestee_name: str
    charges: List[str]
    arrest_date: datetime
    booking_date: Optional[datetime]
    arresting_officer: Optional[str]
    department: Optional[str]
    location: Optional[str]
    severity: Optional[str]  # misdemeanor, felony, etc.
    status: Optional[str]    # active, resolved, etc.
    source: str
    raw_data: Dict
    processed_timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return asdict(self)


class ArrestEventNormalizer:
    """Normalizes arrest event data from various sources."""
    
    def __init__(self):
        """Initialize the normalizer."""
        self.charge_categories = self._load_charge_categories()
        self.officer_name_patterns = self._load_officer_patterns()
        
    def normalize_events(self, raw_events: List[Dict], source: str = "unknown") -> List[NormalizedArrestEvent]:
        """
        Normalize a list of raw arrest events.
        
        Args:
            raw_events: List of raw event dictionaries
            source: Source identifier for the events
            
        Returns:
            List of normalized arrest events
        """
        normalized_events = []
        
        for raw_event in raw_events:
            try:
                normalized_event = self.normalize_single_event(raw_event, source)
                if normalized_event:
                    normalized_events.append(normalized_event)
            except Exception as e:
                logger.error(f"Failed to normalize event {raw_event}: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized_events)} events from {len(raw_events)} raw events")
        return normalized_events
    
    def normalize_single_event(self, raw_event: Dict, source: str = "unknown") -> Optional[NormalizedArrestEvent]:
        """
        Normalize a single arrest event.
        
        Args:
            raw_event: Raw event dictionary
            source: Source identifier
            
        Returns:
            Normalized arrest event or None if normalization fails
        """
        try:
            # Generate unique event ID
            event_id = self._generate_event_id(raw_event)
            
            # Extract and normalize core fields
            arrestee_name = self._normalize_name(raw_event)
            if not arrestee_name:
                logger.warning(f"No valid name found in event: {raw_event}")
                return None
            
            charges = self._normalize_charges(raw_event)
            arrest_date = self._normalize_date(raw_event, 'arrest_date')
            booking_date = self._normalize_date(raw_event, 'booking_date')
            
            # If no arrest date but we have booking date, use booking date
            if not arrest_date and booking_date:
                arrest_date = booking_date
            
            if not arrest_date:
                logger.warning(f"No valid date found in event: {raw_event}")
                return None
            
            # Extract optional fields
            booking_number = self._extract_field(raw_event, ['booking_number', 'booking_id', 'book_num'])
            arresting_officer = self._normalize_officer_name(raw_event)
            department = self._normalize_department(raw_event)
            location = self._extract_field(raw_event, ['location', 'arrest_location', 'jurisdiction'])
            severity = self._determine_severity(charges)
            status = self._normalize_status(raw_event)
            
            # Create normalized event
            normalized_event = NormalizedArrestEvent(
                event_id=event_id,
                booking_number=booking_number,
                arrestee_name=arrestee_name,
                charges=charges,
                arrest_date=arrest_date,
                booking_date=booking_date,
                arresting_officer=arresting_officer,
                department=department,
                location=location,
                severity=severity,
                status=status,
                source=source,
                raw_data=raw_event.copy(),
                processed_timestamp=datetime.now(timezone.utc)
            )
            
            return normalized_event
            
        except Exception as e:
            logger.error(f"Error normalizing event {raw_event}: {e}")
            return None
    
    def _generate_event_id(self, raw_event: Dict) -> str:
        """Generate a unique ID for the event."""
        # Create a hash based on key identifying information
        identifier_parts = []
        
        # Add name if available
        name = self._extract_field(raw_event, ['name', 'arrestee_name', 'defendant_name'])
        if name:
            identifier_parts.append(name.strip().lower())
        
        # Add booking number if available
        booking_num = self._extract_field(raw_event, ['booking_number', 'booking_id'])
        if booking_num:
            identifier_parts.append(str(booking_num).strip())
        
        # Add date if available
        date_str = self._extract_field(raw_event, ['arrest_date', 'booking_date', 'date'])
        if date_str:
            identifier_parts.append(str(date_str).strip())
        
        # Create hash
        identifier_string = "|".join(identifier_parts) or str(raw_event)
        event_hash = hashlib.md5(identifier_string.encode()).hexdigest()[:12]
        
        return f"arrest_{event_hash}"
    
    def _normalize_name(self, raw_event: Dict) -> Optional[str]:
        """Normalize arrestee name."""
        name_fields = ['name', 'arrestee_name', 'defendant_name', 'full_name']
        name = self._extract_field(raw_event, name_fields)
        
        if not name:
            return None
        
        # Clean up the name
        name = re.sub(r'\s+', ' ', str(name).strip())
        name = re.sub(r'[^\w\s\-\',.]', '', name)
        
        # Convert to title case if all caps
        if name.isupper():
            name = name.title()
        
        return name if len(name) > 1 else None
    
    def _normalize_charges(self, raw_event: Dict) -> List[str]:
        """Normalize and categorize charges."""
        charge_fields = ['charge', 'charges', 'offense', 'violations', 'citation']
        charges_raw = self._extract_field(raw_event, charge_fields)
        
        if not charges_raw:
            return []
        
        # Handle multiple charges (separated by various delimiters)
        charges_str = str(charges_raw)
        charge_list = re.split(r'[;,\|]', charges_str)
        
        normalized_charges = []
        for charge in charge_list:
            charge = charge.strip()
            if len(charge) > 2:  # Filter out very short strings
                # Clean up charge text
                charge = re.sub(r'\s+', ' ', charge)
                charge = re.sub(r'^[^\w]*|[^\w]*$', '', charge)  # Remove leading/trailing non-word chars
                normalized_charges.append(charge)
        
        return normalized_charges[:10]  # Limit to prevent excessive data
    
    def _normalize_date(self, raw_event: Dict, date_type: str) -> Optional[datetime]:
        """Normalize date fields."""
        if date_type == 'arrest_date':
            date_fields = ['arrest_date', 'date', 'incident_date']
        elif date_type == 'booking_date':
            date_fields = ['booking_date', 'processed_date', 'intake_date']
        else:
            date_fields = ['date', 'timestamp']
        
        date_str = self._extract_field(raw_event, date_fields)
        
        if not date_str:
            return None
        
        # Try various date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%Y%m%d',
        ]
        
        date_str = str(date_str).strip()
        
        for date_format in date_formats:
            try:
                return datetime.strptime(date_str, date_format)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _normalize_officer_name(self, raw_event: Dict) -> Optional[str]:
        """Normalize officer name."""
        officer_fields = ['officer', 'arresting_officer', 'officer_name', 'badge_name']
        officer = self._extract_field(raw_event, officer_fields)
        
        if not officer:
            return None
        
        officer = str(officer).strip()
        
        # Clean up officer name
        officer = re.sub(r'(Officer|Detective|Sgt|Sergeant|Lt|Lieutenant|Captain|Chief)\s*', '', officer, flags=re.IGNORECASE)
        officer = re.sub(r'\s+', ' ', officer).strip()
        
        # Convert to title case if all caps
        if officer.isupper():
            officer = officer.title()
        
        return officer if len(officer) > 1 else None
    
    def _normalize_department(self, raw_event: Dict) -> Optional[str]:
        """Normalize department/agency name."""
        dept_fields = ['department', 'agency', 'jurisdiction', 'dept', 'pd']
        department = self._extract_field(raw_event, dept_fields)
        
        if not department:
            return None
        
        department = str(department).strip()
        
        # Standardize common department name patterns
        department = re.sub(r'\bPD\b', 'Police Department', department, flags=re.IGNORECASE)
        department = re.sub(r'\bSO\b', 'Sheriff Office', department, flags=re.IGNORECASE)
        department = re.sub(r'\s+', ' ', department).strip()
        
        return department if len(department) > 2 else None
    
    def _normalize_status(self, raw_event: Dict) -> Optional[str]:
        """Normalize case status."""
        status_fields = ['status', 'case_status', 'disposition']
        status = self._extract_field(raw_event, status_fields)
        
        if not status:
            return 'active'  # Default status
        
        status = str(status).lower().strip()
        
        # Map various status terms to standard values
        status_mapping = {
            'active': 'active',
            'open': 'active',
            'pending': 'active',
            'closed': 'resolved',
            'resolved': 'resolved',
            'dismissed': 'resolved',
            'convicted': 'resolved',
            'released': 'released',
            'bonded': 'released'
        }
        
        for key, value in status_mapping.items():
            if key in status:
                return value
        
        return 'active'  # Default if no match
    
    def _determine_severity(self, charges: List[str]) -> Optional[str]:
        """Determine charge severity based on charge text."""
        if not charges:
            return None
        
        charges_text = ' '.join(charges).lower()
        
        # Check for felony indicators
        felony_indicators = ['felony', 'murder', 'robbery', 'burglary', 'assault', 'drug trafficking']
        for indicator in felony_indicators:
            if indicator in charges_text:
                return 'felony'
        
        # Check for misdemeanor indicators
        misdemeanor_indicators = ['misdemeanor', 'petty', 'minor', 'traffic', 'disorderly']
        for indicator in misdemeanor_indicators:
            if indicator in charges_text:
                return 'misdemeanor'
        
        return 'unknown'
    
    def _extract_field(self, raw_event: Dict, field_names: List[str]) -> Optional[str]:
        """Extract field value from raw event using multiple possible field names."""
        for field_name in field_names:
            if field_name in raw_event and raw_event[field_name] is not None:
                value = str(raw_event[field_name]).strip()
                if value and value.lower() not in ['null', 'none', '', 'n/a']:
                    return value
        return None
    
    def _load_charge_categories(self) -> Dict[str, List[str]]:
        """Load charge category mappings."""
        # This would typically load from a configuration file
        return {
            'violent': ['assault', 'battery', 'murder', 'robbery', 'rape'],
            'property': ['burglary', 'theft', 'vandalism', 'arson'],
            'drug': ['possession', 'trafficking', 'distribution', 'manufacturing'],
            'traffic': ['dui', 'speeding', 'reckless driving', 'hit and run'],
            'public_order': ['disorderly conduct', 'public intoxication', 'trespassing']
        }
    
    def _load_officer_patterns(self) -> List[str]:
        """Load officer name patterns for normalization."""
        return [
            r'Officer\s+(.+)',
            r'Detective\s+(.+)',
            r'Sgt\.?\s+(.+)',
            r'Lieutenant\s+(.+)',
            r'(.+),?\s+Badge\s+#?\d+'
        ]


def main():
    """Main function for testing the normalizer."""
    normalizer = ArrestEventNormalizer()
    print("Arrest Event Normalizer initialized")
    
    # Example usage
    sample_event = {
        'name': 'JOHN DOE',
        'charge': 'ASSAULT; BATTERY',
        'arrest_date': '2024-01-15',
        'officer': 'Officer Smith',
        'department': 'City PD'
    }
    
    normalized = normalizer.normalize_single_event(sample_event, 'test')
    if normalized:
        print(f"Normalized event: {normalized.event_id}")
        print(f"Charges: {normalized.charges}")


if __name__ == "__main__":
    main()