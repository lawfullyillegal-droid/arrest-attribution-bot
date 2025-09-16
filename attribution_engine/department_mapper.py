"""
Department Mapper Module

This module handles mapping and standardizing department/agency names across different data sources.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class Department:
    """Department data structure."""
    department_id: str
    official_name: str
    common_name: str
    abbreviations: List[str]
    jurisdiction_type: str  # city, county, state, federal
    location: Optional[str] = None
    active: bool = True
    known_aliases: List[str] = None
    
    def __post_init__(self):
        if self.known_aliases is None:
            self.known_aliases = []


@dataclass
class DepartmentMatch:
    """Department matching result."""
    department: Department
    confidence_score: float
    match_type: str
    original_input: str


class DepartmentMapper:
    """Maps and standardizes department names across data sources."""
    
    def __init__(self):
        """Initialize the department mapper."""
        self.departments_db: Dict[str, Department] = {}
        self.name_index: Dict[str, str] = {}  # normalized_name -> department_id
        self.abbreviation_index: Dict[str, str] = {}  # abbreviation -> department_id
        self.alias_index: Dict[str, str] = {}  # alias -> department_id
        self.similarity_threshold = 0.85
        
        # Load default departments
        self._load_default_departments()
    
    def add_department(self, department: Department) -> str:
        """
        Add a department to the database.
        
        Args:
            department: Department object to add
            
        Returns:
            Department ID
        """
        self.departments_db[department.department_id] = department
        
        # Update indexes
        self._update_indexes(department)
        
        logger.info(f"Added department {department.official_name} with ID {department.department_id}")
        return department.department_id
    
    def _update_indexes(self, department: Department):
        """Update all indexes for a department."""
        # Official name
        normalized_official = self._normalize_name(department.official_name)
        self.name_index[normalized_official] = department.department_id
        
        # Common name
        if department.common_name != department.official_name:
            normalized_common = self._normalize_name(department.common_name)
            self.name_index[normalized_common] = department.department_id
        
        # Abbreviations
        for abbr in department.abbreviations:
            normalized_abbr = self._normalize_name(abbr)
            self.abbreviation_index[normalized_abbr] = department.department_id
        
        # Aliases
        for alias in department.known_aliases:
            normalized_alias = self._normalize_name(alias)
            self.alias_index[normalized_alias] = department.department_id
    
    def map_department(self, department_name: str) -> Optional[DepartmentMatch]:
        """
        Map a department name to a standardized department.
        
        Args:
            department_name: Raw department name to map
            
        Returns:
            DepartmentMatch if found, None otherwise
        """
        if not department_name:
            return None
        
        normalized_input = self._normalize_name(department_name)
        
        # Try exact matches first
        
        # 1. Official/common name match
        if normalized_input in self.name_index:
            dept_id = self.name_index[normalized_input]
            department = self.departments_db[dept_id]
            return DepartmentMatch(
                department=department,
                confidence_score=1.0,
                match_type='exact_name',
                original_input=department_name
            )
        
        # 2. Abbreviation match
        if normalized_input in self.abbreviation_index:
            dept_id = self.abbreviation_index[normalized_input]
            department = self.departments_db[dept_id]
            return DepartmentMatch(
                department=department,
                confidence_score=0.95,
                match_type='abbreviation',
                original_input=department_name
            )
        
        # 3. Alias match
        if normalized_input in self.alias_index:
            dept_id = self.alias_index[normalized_input]
            department = self.departments_db[dept_id]
            return DepartmentMatch(
                department=department,
                confidence_score=0.9,
                match_type='alias',
                original_input=department_name
            )
        
        # 4. Fuzzy matching
        return self._fuzzy_match_department(department_name, normalized_input)
    
    def _fuzzy_match_department(self, original_name: str, normalized_input: str) -> Optional[DepartmentMatch]:
        """Perform fuzzy matching for department names."""
        best_match = None
        best_score = 0
        
        # Check against all indexed names
        all_names = {**self.name_index, **self.abbreviation_index, **self.alias_index}
        
        for indexed_name, dept_id in all_names.items():
            similarity = SequenceMatcher(None, normalized_input, indexed_name).ratio()
            
            if similarity >= self.similarity_threshold and similarity > best_score:
                department = self.departments_db[dept_id]
                best_match = DepartmentMatch(
                    department=department,
                    confidence_score=similarity * 0.8,  # Reduce confidence for fuzzy matches
                    match_type=f'fuzzy_match_{similarity:.2f}',
                    original_input=original_name
                )
                best_score = similarity
        
        return best_match
    
    def _normalize_name(self, name: str) -> str:
        """Normalize department name for matching."""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = str(name).lower().strip()
        
        # Standardize common abbreviations and words
        replacements = {
            r'\bpd\b': 'police department',
            r'\bpolic dept\b': 'police department',
            r'\bso\b': 'sheriff office',
            r'\bsheriff off\b': 'sheriff office',
            r'\bdept\b': 'department',
            r'\bco\b': 'county',
            r'\bcty\b': 'city',
            r'\bmuni\b': 'municipal',
            r'\bmunic\b': 'municipal',
            r'\bmet\b': 'metropolitan',
            r'\bmetro\b': 'metropolitan',
            r'\bst\b': 'state',
            r'\bhwy\b': 'highway',
            r'\btraffic\b': 'traffic',
            r'\bpatrol\b': 'patrol'
        }
        
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove common punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _load_default_departments(self):
        """Load common department configurations."""
        default_departments = [
            {
                'id': 'generic_city_pd',
                'official_name': 'City Police Department',
                'common_name': 'City PD',
                'abbreviations': ['CPD', 'City PD'],
                'jurisdiction_type': 'city',
                'aliases': ['Police Department', 'PD', 'City Police']
            },
            {
                'id': 'generic_county_so',
                'official_name': 'County Sheriff Office',
                'common_name': 'Sheriff Office',
                'abbreviations': ['SO', 'CSO', 'Sheriff'],
                'jurisdiction_type': 'county',
                'aliases': ['Sheriff Department', 'County Sheriff', 'Sheriff Office']
            },
            {
                'id': 'state_police',
                'official_name': 'State Police',
                'common_name': 'State Police',
                'abbreviations': ['SP', 'State PD'],
                'jurisdiction_type': 'state',
                'aliases': ['Highway Patrol', 'State Patrol']
            },
            {
                'id': 'highway_patrol',
                'official_name': 'Highway Patrol',
                'common_name': 'Highway Patrol',
                'abbreviations': ['HP', 'CHP'],
                'jurisdiction_type': 'state',
                'aliases': ['State Highway Patrol', 'Traffic Patrol']
            },
            {
                'id': 'metro_pd',
                'official_name': 'Metropolitan Police Department',
                'common_name': 'Metro PD',
                'abbreviations': ['Metro PD', 'MPD'],
                'jurisdiction_type': 'city',
                'aliases': ['Metro Police', 'Metropolitan Police']
            }
        ]
        
        for dept_data in default_departments:
            department = Department(
                department_id=dept_data['id'],
                official_name=dept_data['official_name'],
                common_name=dept_data['common_name'],
                abbreviations=dept_data['abbreviations'],
                jurisdiction_type=dept_data['jurisdiction_type'],
                known_aliases=dept_data.get('aliases', [])
            )
            self.add_department(department)
    
    def create_department_from_name(self, department_name: str, location: Optional[str] = None) -> Department:
        """
        Create a new department from a name that doesn't match existing departments.
        
        Args:
            department_name: Name of the department
            location: Optional location information
            
        Returns:
            New Department object
        """
        # Generate department ID
        normalized_name = self._normalize_name(department_name)
        dept_id = f"dept_{hash(normalized_name) % 100000:05d}"
        
        # Ensure unique ID
        counter = 1
        base_id = dept_id
        while dept_id in self.departments_db:
            dept_id = f"{base_id}_{counter}"
            counter += 1
        
        # Determine jurisdiction type from name
        jurisdiction_type = self._determine_jurisdiction_type(department_name)
        
        # Generate common name and abbreviations
        common_name = self._generate_common_name(department_name)
        abbreviations = self._generate_abbreviations(department_name)
        
        department = Department(
            department_id=dept_id,
            official_name=department_name.strip(),
            common_name=common_name,
            abbreviations=abbreviations,
            jurisdiction_type=jurisdiction_type,
            location=location,
            known_aliases=[]
        )
        
        return department
    
    def _determine_jurisdiction_type(self, department_name: str) -> str:
        """Determine jurisdiction type from department name."""
        name_lower = department_name.lower()
        
        if any(word in name_lower for word in ['county', 'sheriff']):
            return 'county'
        elif any(word in name_lower for word in ['state', 'highway', 'patrol']):
            return 'state'
        elif any(word in name_lower for word in ['federal', 'fbi', 'dea', 'atf']):
            return 'federal'
        else:
            return 'city'
    
    def _generate_common_name(self, official_name: str) -> str:
        """Generate a common/short name from the official name."""
        name = official_name.strip()
        
        # Replace long forms with short forms
        replacements = [
            ('Police Department', 'PD'),
            ('Sheriff Office', 'SO'),
            ('Sheriff Department', 'SO'),
            ('Highway Patrol', 'HP'),
            ('Metropolitan', 'Metro')
        ]
        
        for long_form, short_form in replacements:
            if long_form in name:
                return name.replace(long_form, short_form)
        
        return name
    
    def _generate_abbreviations(self, department_name: str) -> List[str]:
        """Generate likely abbreviations for a department name."""
        name = department_name.strip()
        abbreviations = []
        
        # Common patterns
        if 'Police Department' in name:
            location = name.replace('Police Department', '').strip()
            if location:
                abbreviations.extend([f'{location} PD', f'{location}PD'])
        
        if 'Sheriff Office' in name or 'Sheriff Department' in name:
            location = re.sub(r'Sheriff (Office|Department)', '', name).strip()
            if location:
                abbreviations.extend([f'{location} SO', f'{location}SO'])
        
        # Generate acronym from words
        words = re.findall(r'\b[A-Z][a-z]*', name)
        if len(words) > 1:
            acronym = ''.join(word[0].upper() for word in words)
            if len(acronym) <= 5:  # Reasonable acronym length
                abbreviations.append(acronym)
        
        return list(set(abbreviations))  # Remove duplicates
    
    def get_department_stats(self) -> Dict:
        """Get statistics about the department database."""
        total_departments = len(self.departments_db)
        active_departments = sum(1 for dept in self.departments_db.values() if dept.active)
        
        jurisdiction_counts = defaultdict(int)
        for dept in self.departments_db.values():
            jurisdiction_counts[dept.jurisdiction_type] += 1
        
        return {
            'total_departments': total_departments,
            'active_departments': active_departments,
            'inactive_departments': total_departments - active_departments,
            'jurisdiction_types': dict(jurisdiction_counts),
            'total_aliases': sum(len(dept.known_aliases) for dept in self.departments_db.values())
        }
    
    def find_similar_departments(self, name: str, limit: int = 5) -> List[Tuple[Department, float]]:
        """
        Find departments with similar names.
        
        Args:
            name: Name to search for
            limit: Maximum number of results
            
        Returns:
            List of (Department, similarity_score) tuples
        """
        normalized_input = self._normalize_name(name)
        similar_departments = []
        
        # Check all departments
        for dept in self.departments_db.values():
            # Check official name
            official_similarity = SequenceMatcher(None, normalized_input, 
                                                self._normalize_name(dept.official_name)).ratio()
            
            # Check common name
            common_similarity = SequenceMatcher(None, normalized_input, 
                                              self._normalize_name(dept.common_name)).ratio()
            
            # Use best similarity
            best_similarity = max(official_similarity, common_similarity)
            
            if best_similarity > 0.3:  # Low threshold for search
                similar_departments.append((dept, best_similarity))
        
        # Sort by similarity and return top results
        similar_departments.sort(key=lambda x: x[1], reverse=True)
        return similar_departments[:limit]
    
    def export_department_mappings(self) -> Dict:
        """Export department mappings for backup or analysis."""
        export_data = {}
        
        for dept_id, dept in self.departments_db.items():
            export_data[dept_id] = {
                'official_name': dept.official_name,
                'common_name': dept.common_name,
                'abbreviations': dept.abbreviations,
                'jurisdiction_type': dept.jurisdiction_type,
                'location': dept.location,
                'active': dept.active,
                'known_aliases': dept.known_aliases
            }
        
        return export_data


def main():
    """Main function for testing the department mapper."""
    mapper = DepartmentMapper()
    
    # Test mapping
    test_names = [
        'NYC PD',
        'Los Angeles Police Department',
        'County Sheriff',
        'State Police',
        'Metro Police'
    ]
    
    print("Testing department mapping:")
    for name in test_names:
        match = mapper.map_department(name)
        if match:
            print(f"'{name}' -> '{match.department.official_name}' "
                  f"(confidence: {match.confidence_score:.2f}, type: {match.match_type})")
        else:
            print(f"'{name}' -> No match found")
    
    # Print stats
    stats = mapper.get_department_stats()
    print(f"\nDepartment database stats: {stats}")


if __name__ == "__main__":
    main()