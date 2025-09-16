"""
Timestamp Verifier Module

This module handles verification and validation of timestamps in arrest and booking data.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Union
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimestampValidationResult(Enum):
    """Timestamp validation result codes."""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    FUTURE_DATE = "future_date"
    TOO_OLD = "too_old"
    INVALID_TIME = "invalid_time"
    MISSING_TIMEZONE = "missing_timezone"
    INCONSISTENT_SEQUENCE = "inconsistent_sequence"


@dataclass
class TimestampValidation:
    """Timestamp validation result."""
    original_value: str
    parsed_datetime: Optional[datetime]
    result: TimestampValidationResult
    confidence_score: float
    notes: List[str]
    suggested_correction: Optional[datetime] = None


@dataclass
class EventTimestampSequence:
    """Represents a sequence of timestamps for an arrest event."""
    incident_time: Optional[datetime] = None
    arrest_time: Optional[datetime] = None
    booking_time: Optional[datetime] = None
    processing_time: Optional[datetime] = None
    
    def is_valid_sequence(self) -> bool:
        """Check if the timestamp sequence is logically valid."""
        timestamps = [t for t in [self.incident_time, self.arrest_time, 
                                 self.booking_time, self.processing_time] if t is not None]
        
        # Check if timestamps are in chronological order
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                return False
        
        return True


class TimestampVerifier:
    """Verifies and validates timestamps in arrest data."""
    
    def __init__(self, max_age_days: int = 3650, future_tolerance_hours: int = 24):
        """
        Initialize the timestamp verifier.
        
        Args:
            max_age_days: Maximum age in days for valid timestamps
            future_tolerance_hours: Tolerance for future timestamps in hours
        """
        self.max_age_days = max_age_days
        self.future_tolerance_hours = future_tolerance_hours
        self.common_formats = self._get_common_formats()
        
    def verify_timestamp(self, timestamp_value: Union[str, datetime], 
                        context: str = "unknown") -> TimestampValidation:
        """
        Verify a single timestamp value.
        
        Args:
            timestamp_value: Timestamp to verify (string or datetime)
            context: Context of the timestamp (e.g., 'arrest_time', 'booking_time')
            
        Returns:
            TimestampValidation result
        """
        original_str = str(timestamp_value) if timestamp_value is not None else ""
        notes = []
        
        # Handle already parsed datetime
        if isinstance(timestamp_value, datetime):
            return self._validate_datetime(timestamp_value, original_str, context, notes)
        
        # Handle string input
        if not timestamp_value or not str(timestamp_value).strip():
            return TimestampValidation(
                original_value=original_str,
                parsed_datetime=None,
                result=TimestampValidationResult.INVALID_FORMAT,
                confidence_score=0.0,
                notes=["Empty or null timestamp"]
            )
        
        # Try to parse the timestamp
        parsed_dt, parse_confidence, parse_notes = self._parse_timestamp(str(timestamp_value))
        notes.extend(parse_notes)
        
        if parsed_dt is None:
            return TimestampValidation(
                original_value=original_str,
                parsed_datetime=None,
                result=TimestampValidationResult.INVALID_FORMAT,
                confidence_score=0.0,
                notes=notes
            )
        
        # Validate the parsed datetime
        return self._validate_datetime(parsed_dt, original_str, context, notes)
    
    def _parse_timestamp(self, timestamp_str: str) -> Tuple[Optional[datetime], float, List[str]]:
        """
        Parse timestamp string using various formats.
        
        Returns:
            Tuple of (parsed_datetime, confidence_score, notes)
        """
        timestamp_str = timestamp_str.strip()
        notes = []
        
        # Try each format
        for fmt, confidence in self.common_formats:
            try:
                parsed_dt = datetime.strptime(timestamp_str, fmt)
                
                # Add timezone info if missing (assume local time)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                    notes.append(f"Assumed UTC timezone for format {fmt}")
                
                notes.append(f"Parsed using format: {fmt}")
                return parsed_dt, confidence, notes
                
            except ValueError:
                continue
        
        # Try ISO format parsing (more flexible)
        try:
            parsed_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            notes.append("Parsed using ISO format")
            return parsed_dt, 0.9, notes
        except ValueError:
            pass
        
        # Try parsing with regex for common patterns
        parsed_dt = self._parse_with_regex(timestamp_str)
        if parsed_dt:
            notes.append("Parsed using regex pattern matching")
            return parsed_dt, 0.7, notes
        
        notes.append(f"Failed to parse: {timestamp_str}")
        return None, 0.0, notes
    
    def _parse_with_regex(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp using regex patterns for non-standard formats."""
        
        # Pattern: MM/DD/YYYY HH:MM (AM/PM)
        pattern1 = r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?:\s*(AM|PM))?'
        match1 = re.search(pattern1, timestamp_str, re.IGNORECASE)
        if match1:
            month, day, year, hour, minute = map(int, match1.groups()[:5])
            am_pm = match1.group(6)
            
            if am_pm and am_pm.upper() == 'PM' and hour != 12:
                hour += 12
            elif am_pm and am_pm.upper() == 'AM' and hour == 12:
                hour = 0
            
            try:
                return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                pass
        
        # Pattern: YYYY-MM-DD HH:MM
        pattern2 = r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})'
        match2 = re.search(pattern2, timestamp_str)
        if match2:
            year, month, day, hour, minute = map(int, match2.groups())
            try:
                return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                pass
        
        # Pattern: DD-MM-YYYY HH:MM
        pattern3 = r'(\d{1,2})-(\d{1,2})-(\d{4})\s+(\d{1,2}):(\d{2})'
        match3 = re.search(pattern3, timestamp_str)
        if match3:
            day, month, year, hour, minute = map(int, match3.groups())
            try:
                return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                pass
        
        return None
    
    def _validate_datetime(self, dt: datetime, original_str: str, 
                          context: str, notes: List[str]) -> TimestampValidation:
        """Validate a parsed datetime object."""
        now = datetime.now(timezone.utc)
        
        # Ensure dt has timezone info for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        confidence = 1.0
        
        # Check if date is too far in the future
        future_limit = now + timedelta(hours=self.future_tolerance_hours)
        if dt > future_limit:
            return TimestampValidation(
                original_value=original_str,
                parsed_datetime=dt,
                result=TimestampValidationResult.FUTURE_DATE,
                confidence_score=0.0,
                notes=notes + [f"Date is {(dt - now).days} days in the future"],
                suggested_correction=now
            )
        
        # Check if date is too old
        oldest_valid = now - timedelta(days=self.max_age_days)
        if dt < oldest_valid:
            return TimestampValidation(
                original_value=original_str,
                parsed_datetime=dt,
                result=TimestampValidationResult.TOO_OLD,
                confidence_score=0.2,
                notes=notes + [f"Date is {(now - dt).days} days old"]
            )
        
        # Check for reasonable time (business hours vs odd hours)
        if dt.hour < 6 or dt.hour > 23:
            confidence *= 0.9
            notes.append(f"Unusual hour: {dt.hour}")
        
        # Reduce confidence for very old dates
        age_days = (now - dt).days
        if age_days > 365:  # More than a year old
            confidence *= 0.8
            notes.append(f"Date is {age_days} days old")
        
        # Check for timezone issues
        if dt.tzinfo is None:
            confidence *= 0.7
            notes.append("No timezone information")
        
        return TimestampValidation(
            original_value=original_str,
            parsed_datetime=dt,
            result=TimestampValidationResult.VALID,
            confidence_score=confidence,
            notes=notes
        )
    
    def verify_event_sequence(self, event_data: Dict) -> Tuple[EventTimestampSequence, List[str]]:
        """
        Verify timestamp sequence for an entire arrest event.
        
        Args:
            event_data: Dictionary containing timestamp fields
            
        Returns:
            Tuple of (EventTimestampSequence, validation_notes)
        """
        notes = []
        sequence = EventTimestampSequence()
        
        # Extract and verify each timestamp
        timestamp_fields = {
            'incident_time': ['incident_time', 'offense_time', 'crime_time'],
            'arrest_time': ['arrest_time', 'arrest_date', 'arrest_timestamp'],
            'booking_time': ['booking_time', 'booking_date', 'intake_time'],
            'processing_time': ['processing_time', 'processed_date', 'completed_time']
        }
        
        for field_name, possible_keys in timestamp_fields.items():
            timestamp_value = None
            
            # Find the timestamp value
            for key in possible_keys:
                if key in event_data and event_data[key]:
                    timestamp_value = event_data[key]
                    break
            
            if timestamp_value:
                validation = self.verify_timestamp(timestamp_value, field_name)
                
                if validation.result == TimestampValidationResult.VALID:
                    setattr(sequence, field_name, validation.parsed_datetime)
                    if validation.confidence_score < 0.8:
                        notes.append(f"{field_name}: Low confidence ({validation.confidence_score:.2f})")
                else:
                    notes.append(f"{field_name}: {validation.result.value} - {validation.notes}")
        
        # Validate sequence logic
        sequence_notes = self._validate_sequence_logic(sequence)
        notes.extend(sequence_notes)
        
        return sequence, notes
    
    def _validate_sequence_logic(self, sequence: EventTimestampSequence) -> List[str]:
        """Validate the logical consistency of timestamp sequence."""
        notes = []
        
        timestamps = [
            ('incident_time', sequence.incident_time),
            ('arrest_time', sequence.arrest_time),
            ('booking_time', sequence.booking_time),
            ('processing_time', sequence.processing_time)
        ]
        
        # Remove None values and sort by time
        valid_timestamps = [(name, dt) for name, dt in timestamps if dt is not None]
        
        if len(valid_timestamps) < 2:
            return notes  # Not enough timestamps to validate sequence
        
        # Check chronological order
        for i in range(1, len(valid_timestamps)):
            prev_name, prev_dt = valid_timestamps[i-1]
            curr_name, curr_dt = valid_timestamps[i]
            
            if curr_dt < prev_dt:
                notes.append(f"Timestamp sequence error: {curr_name} ({curr_dt}) is before {prev_name} ({prev_dt})")
            
            # Check reasonable time gaps
            time_diff = curr_dt - prev_dt
            
            # Arrest to booking should typically be within 24 hours
            if prev_name == 'arrest_time' and curr_name == 'booking_time':
                if time_diff > timedelta(hours=48):
                    notes.append(f"Long delay between arrest and booking: {time_diff}")
                elif time_diff < timedelta(minutes=5):
                    notes.append(f"Very short time between arrest and booking: {time_diff}")
            
            # Booking to processing should be reasonable
            if prev_name == 'booking_time' and curr_name == 'processing_time':
                if time_diff > timedelta(hours=72):
                    notes.append(f"Long delay between booking and processing: {time_diff}")
        
        return notes
    
    def _get_common_formats(self) -> List[Tuple[str, float]]:
        """Get common timestamp formats with confidence scores."""
        return [
            # ISO formats (high confidence)
            ('%Y-%m-%d %H:%M:%S', 0.95),
            ('%Y-%m-%dT%H:%M:%S', 0.95),
            ('%Y-%m-%dT%H:%M:%SZ', 0.95),
            ('%Y-%m-%dT%H:%M:%S%z', 0.95),
            
            # US formats
            ('%m/%d/%Y %H:%M:%S', 0.9),
            ('%m/%d/%Y %I:%M:%S %p', 0.9),
            ('%m/%d/%Y %H:%M', 0.85),
            ('%m/%d/%Y %I:%M %p', 0.85),
            ('%m/%d/%Y', 0.8),
            
            # European formats
            ('%d/%m/%Y %H:%M:%S', 0.8),
            ('%d/%m/%Y %H:%M', 0.75),
            ('%d/%m/%Y', 0.7),
            
            # Other common formats
            ('%Y%m%d %H%M%S', 0.7),
            ('%Y%m%d', 0.65),
            ('%m-%d-%Y %H:%M:%S', 0.85),
            ('%m-%d-%Y %H:%M', 0.8),
            ('%m-%d-%Y', 0.75),
        ]
    
    def get_validation_stats(self, validations: List[TimestampValidation]) -> Dict:
        """Get statistics from a list of timestamp validations."""
        if not validations:
            return {}
        
        total = len(validations)
        valid_count = sum(1 for v in validations if v.result == TimestampValidationResult.VALID)
        
        result_counts = {}
        for validation in validations:
            result_counts[validation.result.value] = result_counts.get(validation.result.value, 0) + 1
        
        avg_confidence = sum(v.confidence_score for v in validations) / total
        
        return {
            'total_timestamps': total,
            'valid_count': valid_count,
            'valid_percentage': (valid_count / total) * 100,
            'average_confidence': avg_confidence,
            'result_breakdown': result_counts
        }


def main():
    """Main function for testing the timestamp verifier."""
    verifier = TimestampVerifier()
    
    # Test various timestamp formats
    test_timestamps = [
        '2024-01-15 14:30:00',
        '01/15/2024 2:30 PM',
        '15/01/2024 14:30',
        '2024-01-15T14:30:00Z',
        '2025-12-31 23:59:59',  # Future date
        '1990-01-01 12:00:00',  # Old date
        'invalid timestamp',
        ''
    ]
    
    print("Testing timestamp verification:")
    validations = []
    
    for timestamp in test_timestamps:
        validation = verifier.verify_timestamp(timestamp)
        validations.append(validation)
        
        print(f"'{timestamp}' -> {validation.result.value} "
              f"(confidence: {validation.confidence_score:.2f})")
        if validation.notes:
            print(f"  Notes: {', '.join(validation.notes)}")
        print()
    
    # Test event sequence
    print("Testing event sequence:")
    event_data = {
        'arrest_time': '2024-01-15 14:30:00',
        'booking_time': '2024-01-15 16:45:00',
        'processing_time': '2024-01-15 18:20:00'
    }
    
    sequence, notes = verifier.verify_event_sequence(event_data)
    print(f"Sequence valid: {sequence.is_valid_sequence()}")
    if notes:
        print(f"Notes: {', '.join(notes)}")
    
    # Print validation stats
    stats = verifier.get_validation_stats(validations)
    print(f"\nValidation statistics: {stats}")


if __name__ == "__main__":
    main()