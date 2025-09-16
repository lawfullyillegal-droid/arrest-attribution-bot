"""
Officer Matcher Module

This module handles matching and identifying officers across different data sources.
"""

from typing import List, Dict, Optional, Tuple, Set
import re
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Officer:
    """Officer data structure."""
    officer_id: str
    name: str
    badge_number: Optional[str] = None
    department: Optional[str] = None
    known_aliases: List[str] = None
    active_status: bool = True
    
    def __post_init__(self):
        if self.known_aliases is None:
            self.known_aliases = []


@dataclass
class OfficerMatch:
    """Officer matching result."""
    officer: Officer
    confidence_score: float
    match_criteria: List[str]
    source_data: Dict


class OfficerMatcher:
    """Matches officers across different data sources and maintains officer profiles."""
    
    def __init__(self):
        """Initialize the officer matcher."""
        self.officers_db: Dict[str, Officer] = {}
        self.name_index: Dict[str, Set[str]] = defaultdict(set)  # normalized_name -> officer_ids
        self.badge_index: Dict[str, str] = {}  # badge_number -> officer_id
        self.similarity_threshold = 0.85
        
    def add_officer(self, officer: Officer) -> str:
        """
        Add an officer to the database.
        
        Args:
            officer: Officer object to add
            
        Returns:
            Officer ID
        """
        self.officers_db[officer.officer_id] = officer
        
        # Update name index
        normalized_name = self._normalize_name(officer.name)
        self.name_index[normalized_name].add(officer.officer_id)
        
        # Add aliases to name index
        for alias in officer.known_aliases:
            normalized_alias = self._normalize_name(alias)
            self.name_index[normalized_alias].add(officer.officer_id)
        
        # Update badge index
        if officer.badge_number:
            self.badge_index[officer.badge_number] = officer.officer_id
        
        logger.info(f"Added officer {officer.name} with ID {officer.officer_id}")
        return officer.officer_id
    
    def match_officer(self, officer_data: Dict) -> Optional[OfficerMatch]:
        """
        Match officer data against known officers.
        
        Args:
            officer_data: Dictionary containing officer information
            
        Returns:
            OfficerMatch if found, None otherwise
        """
        name = officer_data.get('name') or officer_data.get('officer_name') or officer_data.get('arresting_officer')
        badge_number = officer_data.get('badge_number') or officer_data.get('badge')
        department = officer_data.get('department') or officer_data.get('agency')
        
        if not name and not badge_number:
            logger.warning("No name or badge number provided for officer matching")
            return None
        
        # Try exact badge match first
        if badge_number and badge_number in self.badge_index:
            officer_id = self.badge_index[badge_number]
            officer = self.officers_db[officer_id]
            return OfficerMatch(
                officer=officer,
                confidence_score=1.0,
                match_criteria=['exact_badge_match'],
                source_data=officer_data
            )
        
        # Try name-based matching
        if name:
            return self._match_by_name(name, department, officer_data)
        
        return None
    
    def _match_by_name(self, name: str, department: Optional[str], source_data: Dict) -> Optional[OfficerMatch]:
        """Match officer by name with fuzzy matching."""
        normalized_name = self._normalize_name(name)
        
        # Try exact name match
        if normalized_name in self.name_index:
            candidate_ids = self.name_index[normalized_name]
            best_match = self._select_best_candidate(candidate_ids, department, source_data)
            if best_match:
                return best_match
        
        # Try fuzzy name matching
        best_match = None
        best_score = 0
        
        for indexed_name, officer_ids in self.name_index.items():
            similarity = SequenceMatcher(None, normalized_name, indexed_name).ratio()
            
            if similarity >= self.similarity_threshold and similarity > best_score:
                candidate_match = self._select_best_candidate(officer_ids, department, source_data)
                if candidate_match and candidate_match.confidence_score > best_score:
                    best_match = candidate_match
                    best_score = similarity
                    # Adjust confidence based on name similarity
                    best_match.confidence_score = similarity * candidate_match.confidence_score
                    best_match.match_criteria.append(f'fuzzy_name_match_{similarity:.2f}')
        
        return best_match
    
    def _select_best_candidate(self, candidate_ids: Set[str], department: Optional[str], source_data: Dict) -> Optional[OfficerMatch]:
        """Select the best candidate from multiple matches."""
        if not candidate_ids:
            return None
        
        if len(candidate_ids) == 1:
            officer_id = next(iter(candidate_ids))
            officer = self.officers_db[officer_id]
            confidence = self._calculate_confidence(officer, department, source_data)
            match_criteria = ['exact_name_match']
            
            if department and officer.department:
                if self._normalize_department(department) == self._normalize_department(officer.department):
                    match_criteria.append('department_match')
                else:
                    confidence *= 0.8  # Reduce confidence for department mismatch
                    match_criteria.append('department_mismatch')
            
            return OfficerMatch(
                officer=officer,
                confidence_score=confidence,
                match_criteria=match_criteria,
                source_data=source_data
            )
        
        # Multiple candidates - select best based on additional criteria
        best_officer = None
        best_confidence = 0
        best_criteria = []
        
        for officer_id in candidate_ids:
            officer = self.officers_db[officer_id]
            confidence = self._calculate_confidence(officer, department, source_data)
            criteria = ['exact_name_match']
            
            # Boost confidence for department match
            if department and officer.department:
                if self._normalize_department(department) == self._normalize_department(officer.department):
                    confidence *= 1.2
                    criteria.append('department_match')
                else:
                    confidence *= 0.8
                    criteria.append('department_mismatch')
            
            # Boost confidence for active officers
            if officer.active_status:
                confidence *= 1.1
                criteria.append('active_officer')
            
            if confidence > best_confidence:
                best_officer = officer
                best_confidence = confidence
                best_criteria = criteria
        
        if best_officer:
            return OfficerMatch(
                officer=best_officer,
                confidence_score=min(best_confidence, 1.0),  # Cap at 1.0
                match_criteria=best_criteria,
                source_data=source_data
            )
        
        return None
    
    def _calculate_confidence(self, officer: Officer, department: Optional[str], source_data: Dict) -> float:
        """Calculate confidence score for an officer match."""
        base_confidence = 0.9
        
        # Adjust based on available information
        if officer.badge_number:
            base_confidence += 0.05
        
        if officer.department and department:
            dept_match = self._normalize_department(officer.department) == self._normalize_department(department)
            if dept_match:
                base_confidence += 0.05
            else:
                base_confidence -= 0.1
        
        if officer.active_status:
            base_confidence += 0.02
        
        return min(base_confidence, 1.0)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize officer name for matching."""
        if not name:
            return ""
        
        # Convert to lowercase and remove common prefixes/suffixes
        name = str(name).lower().strip()
        
        # Remove titles and ranks
        name = re.sub(r'\b(officer|detective|sgt|sergeant|lt|lieutenant|captain|chief|deputy)\b', '', name)
        
        # Remove extra whitespace and punctuation
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _normalize_department(self, department: str) -> str:
        """Normalize department name for comparison."""
        if not department:
            return ""
        
        dept = str(department).lower().strip()
        
        # Standardize common abbreviations
        dept = re.sub(r'\bpd\b', 'police department', dept)
        dept = re.sub(r'\bso\b', 'sheriff office', dept)
        dept = re.sub(r'\bdept\b', 'department', dept)
        
        # Remove extra whitespace
        dept = re.sub(r'\s+', ' ', dept).strip()
        
        return dept
    
    def create_officer_from_data(self, officer_data: Dict) -> Officer:
        """
        Create a new Officer object from raw data.
        
        Args:
            officer_data: Dictionary containing officer information
            
        Returns:
            New Officer object
        """
        name = officer_data.get('name') or officer_data.get('officer_name') or officer_data.get('arresting_officer')
        if not name:
            raise ValueError("Officer name is required")
        
        # Generate officer ID
        normalized_name = self._normalize_name(name)
        officer_id = f"officer_{hash(normalized_name) % 100000:05d}"
        
        # Ensure unique ID
        counter = 1
        base_id = officer_id
        while officer_id in self.officers_db:
            officer_id = f"{base_id}_{counter}"
            counter += 1
        
        officer = Officer(
            officer_id=officer_id,
            name=name.strip(),
            badge_number=officer_data.get('badge_number'),
            department=officer_data.get('department'),
            known_aliases=[],
            active_status=True
        )
        
        return officer
    
    def get_officer_stats(self) -> Dict:
        """Get statistics about the officer database."""
        total_officers = len(self.officers_db)
        active_officers = sum(1 for officer in self.officers_db.values() if officer.active_status)
        departments = set(officer.department for officer in self.officers_db.values() if officer.department)
        
        return {
            'total_officers': total_officers,
            'active_officers': active_officers,
            'inactive_officers': total_officers - active_officers,
            'departments': len(departments),
            'officers_with_badges': sum(1 for officer in self.officers_db.values() if officer.badge_number)
        }
    
    def find_similar_officers(self, name: str, limit: int = 5) -> List[Tuple[Officer, float]]:
        """
        Find officers with similar names.
        
        Args:
            name: Name to search for
            limit: Maximum number of results
            
        Returns:
            List of (Officer, similarity_score) tuples
        """
        normalized_name = self._normalize_name(name)
        similar_officers = []
        
        for indexed_name, officer_ids in self.name_index.items():
            similarity = SequenceMatcher(None, normalized_name, indexed_name).ratio()
            
            if similarity > 0.5:  # Lower threshold for search
                for officer_id in officer_ids:
                    officer = self.officers_db[officer_id]
                    similar_officers.append((officer, similarity))
        
        # Sort by similarity and return top results
        similar_officers.sort(key=lambda x: x[1], reverse=True)
        return similar_officers[:limit]


def main():
    """Main function for testing the officer matcher."""
    matcher = OfficerMatcher()
    
    # Create sample officers
    officer1 = Officer(
        officer_id="officer_001",
        name="John Smith",
        badge_number="12345",
        department="City Police Department"
    )
    
    officer2 = Officer(
        officer_id="officer_002",
        name="Jane Doe",
        badge_number="67890",
        department="County Sheriff Office"
    )
    
    # Add officers to matcher
    matcher.add_officer(officer1)
    matcher.add_officer(officer2)
    
    # Test matching
    test_data = {
        'name': 'J. Smith',
        'department': 'City PD'
    }
    
    match = matcher.match_officer(test_data)
    if match:
        print(f"Matched officer: {match.officer.name} (confidence: {match.confidence_score:.2f})")
        print(f"Match criteria: {match.match_criteria}")
    else:
        print("No match found")
    
    # Print stats
    stats = matcher.get_officer_stats()
    print(f"Officer database stats: {stats}")


if __name__ == "__main__":
    main()