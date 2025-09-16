"""
Geo Locator Module

This module provides geocoding and location-based utilities for arrest attribution.
"""

from typing import Optional, Dict, List, Tuple, Union
import re
import logging
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Location data structure."""
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    county: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[str] = None  # high, medium, low
    source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'county': self.county,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'source': self.source
        }
    
    def is_complete(self) -> bool:
        """Check if location has sufficient information."""
        return bool(self.city and self.state) or bool(self.latitude and self.longitude)


@dataclass
class JurisdictionBoundary:
    """Jurisdiction boundary definition."""
    jurisdiction_id: str
    name: str
    jurisdiction_type: str  # city, county, state
    boundaries: List[Tuple[float, float]]  # List of (lat, lng) points
    center_point: Tuple[float, float]  # (lat, lng) of center
    
    def contains_point(self, latitude: float, longitude: float) -> bool:
        """Check if a point is within this jurisdiction using ray casting algorithm."""
        if not self.boundaries:
            return False
        
        x, y = longitude, latitude
        n = len(self.boundaries)
        inside = False
        
        p1x, p1y = self.boundaries[0]
        for i in range(1, n + 1):
            p2x, p2y = self.boundaries[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class GeoLocator:
    """Provides location services and geocoding for arrest attribution."""
    
    def __init__(self, jurisdictions_file: Optional[str] = None):
        """
        Initialize the geo locator.
        
        Args:
            jurisdictions_file: Path to file containing jurisdiction boundaries
        """
        self.jurisdictions: Dict[str, JurisdictionBoundary] = {}
        self.location_cache: Dict[str, Location] = {}
        self.state_abbreviations = self._load_state_abbreviations()
        
        if jurisdictions_file:
            self._load_jurisdictions(jurisdictions_file)
    
    def parse_address(self, address_string: str) -> Location:
        """
        Parse an address string into components.
        
        Args:
            address_string: Raw address string
            
        Returns:
            Location object with parsed components
        """
        if not address_string:
            return Location()
        
        # Check cache first
        cache_key = address_string.strip().lower()
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
        
        address_string = address_string.strip()
        location = Location(source='parsed')
        
        # Parse using various patterns
        location = self._parse_full_address(address_string, location)
        
        # Cache the result
        self.location_cache[cache_key] = location
        
        return location
    
    def _parse_full_address(self, address: str, location: Location) -> Location:
        """Parse a full address string."""
        
        # Pattern for full address: "123 Main St, Springfield, IL 62701"
        full_pattern = r'^(.+?),\s*([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)?\s*$'
        match = re.match(full_pattern, address.strip(), re.IGNORECASE)
        
        if match:
            location.address = match.group(1).strip()
            location.city = match.group(2).strip()
            location.state = self._normalize_state(match.group(3).strip())
            if match.group(4):
                location.zip_code = match.group(4).strip()
            location.accuracy = 'high'
            return location
        
        # Pattern for city, state: "Springfield, IL"
        city_state_pattern = r'^([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)?\s*$'
        match = re.match(city_state_pattern, address.strip(), re.IGNORECASE)
        
        if match:
            location.city = match.group(1).strip()
            location.state = self._normalize_state(match.group(2).strip())
            if match.group(3):
                location.zip_code = match.group(3).strip()
            location.accuracy = 'medium'
            return location
        
        # Try to extract components from less structured text
        location = self._extract_address_components(address, location)
        
        return location
    
    def _extract_address_components(self, address: str, location: Location) -> Location:
        """Extract address components from unstructured text."""
        
        # Look for ZIP code
        zip_pattern = r'\b(\d{5}(?:-\d{4})?)\b'
        zip_match = re.search(zip_pattern, address)
        if zip_match:
            location.zip_code = zip_match.group(1)
            # Remove ZIP from address for further parsing
            address = re.sub(zip_pattern, '', address).strip()
        
        # Look for state abbreviation
        state_pattern = r'\b([A-Z]{2})\b'
        state_matches = re.findall(state_pattern, address)
        for state_abbr in state_matches:
            if state_abbr in self.state_abbreviations:
                location.state = state_abbr
                # Remove state from address
                address = re.sub(r'\b' + state_abbr + r'\b', '', address).strip()
                break
        
        # What's left might be city or address
        remaining = address.strip(' ,')
        if remaining:
            # If it looks like a street address (contains numbers), treat as address
            if re.search(r'\b\d+\b', remaining):
                location.address = remaining
                location.accuracy = 'low'
            else:
                # Otherwise treat as city
                location.city = remaining
                location.accuracy = 'medium' if location.state else 'low'
        
        return location
    
    def geocode_location(self, location: Location) -> Location:
        """
        Geocode a location to get latitude/longitude.
        Note: This is a placeholder implementation.
        In a real system, this would call a geocoding service.
        
        Args:
            location: Location to geocode
            
        Returns:
            Location with coordinates added
        """
        # This is a mock implementation using approximate coordinates
        # In production, you would use a geocoding service like Google Maps API
        
        if location.latitude and location.longitude:
            return location  # Already geocoded
        
        # Mock geocoding based on city/state
        mock_coordinates = self._get_mock_coordinates(location.city, location.state)
        
        if mock_coordinates:
            location.latitude = mock_coordinates[0]
            location.longitude = mock_coordinates[1]
            location.accuracy = 'medium' if location.address else 'low'
        
        return location
    
    def _get_mock_coordinates(self, city: Optional[str], state: Optional[str]) -> Optional[Tuple[float, float]]:
        """Get mock coordinates for testing purposes."""
        
        # Mock coordinate database for common cities
        mock_coords = {
            ('chicago', 'il'): (41.8781, -87.6298),
            ('new york', 'ny'): (40.7128, -74.0060),
            ('los angeles', 'ca'): (34.0522, -118.2437),
            ('houston', 'tx'): (29.7604, -95.3698),
            ('phoenix', 'az'): (33.4484, -112.0740),
            ('philadelphia', 'pa'): (39.9526, -75.1652),
            ('san antonio', 'tx'): (29.4241, -98.4936),
            ('san diego', 'ca'): (32.7157, -117.1611),
            ('dallas', 'tx'): (32.7767, -96.7970),
            ('san jose', 'ca'): (37.3382, -121.8863),
            ('springfield', 'il'): (39.7817, -89.6501),
            ('springfield', 'mo'): (37.2153, -93.2982)
        }
        
        if city and state:
            key = (city.lower().strip(), state.lower().strip())
            return mock_coords.get(key)
        
        return None
    
    def calculate_distance(self, loc1: Location, loc2: Location) -> Optional[float]:
        """
        Calculate distance between two locations in miles.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in miles, or None if coordinates missing
        """
        if not all([loc1.latitude, loc1.longitude, loc2.latitude, loc2.longitude]):
            return None
        
        return self._haversine_distance(
            loc1.latitude, loc1.longitude,
            loc2.latitude, loc2.longitude
        )
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in miles
        r = 3956
        
        return c * r
    
    def find_jurisdiction(self, location: Location) -> Optional[JurisdictionBoundary]:
        """
        Find the jurisdiction that contains a location.
        
        Args:
            location: Location to check
            
        Returns:
            JurisdictionBoundary if found, None otherwise
        """
        if not (location.latitude and location.longitude):
            return None
        
        for jurisdiction in self.jurisdictions.values():
            if jurisdiction.contains_point(location.latitude, location.longitude):
                return jurisdiction
        
        return None
    
    def get_nearby_jurisdictions(self, location: Location, radius_miles: float = 10) -> List[Tuple[JurisdictionBoundary, float]]:
        """
        Get jurisdictions within a certain radius of a location.
        
        Args:
            location: Center location
            radius_miles: Search radius in miles
            
        Returns:
            List of (JurisdictionBoundary, distance) tuples
        """
        if not (location.latitude and location.longitude):
            return []
        
        nearby = []
        
        for jurisdiction in self.jurisdictions.values():
            center_location = Location(
                latitude=jurisdiction.center_point[0],
                longitude=jurisdiction.center_point[1]
            )
            
            distance = self.calculate_distance(location, center_location)
            
            if distance is not None and distance <= radius_miles:
                nearby.append((jurisdiction, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        
        return nearby
    
    def _load_jurisdictions(self, jurisdictions_file: str):
        """Load jurisdiction boundaries from file."""
        try:
            jurisdictions_path = Path(jurisdictions_file)
            if jurisdictions_path.exists():
                with open(jurisdictions_path, 'r') as f:
                    data = json.load(f)
                
                for jurisdiction_data in data:
                    jurisdiction = JurisdictionBoundary(
                        jurisdiction_id=jurisdiction_data['id'],
                        name=jurisdiction_data['name'],
                        jurisdiction_type=jurisdiction_data['type'],
                        boundaries=[(p['lat'], p['lng']) for p in jurisdiction_data['boundaries']],
                        center_point=(jurisdiction_data['center']['lat'], jurisdiction_data['center']['lng'])
                    )
                    
                    self.jurisdictions[jurisdiction.jurisdiction_id] = jurisdiction
                
                logger.info(f"Loaded {len(self.jurisdictions)} jurisdictions from {jurisdictions_file}")
            
        except Exception as e:
            logger.error(f"Failed to load jurisdictions from {jurisdictions_file}: {e}")
    
    def _load_state_abbreviations(self) -> Dict[str, str]:
        """Load US state abbreviations."""
        return {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }
    
    def _normalize_state(self, state: str) -> str:
        """Normalize state name or abbreviation."""
        state = state.strip().upper()
        
        if len(state) == 2 and state in self.state_abbreviations:
            return state
        
        # Look for full state name
        for abbr, full_name in self.state_abbreviations.items():
            if full_name.upper() == state:
                return abbr
        
        return state  # Return as-is if not found
    
    def validate_location(self, location: Location) -> List[str]:
        """
        Validate location data and return list of issues.
        
        Args:
            location: Location to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for basic completeness
        if not location.is_complete():
            issues.append("Location lacks sufficient information (city/state or coordinates)")
        
        # Validate coordinates if present
        if location.latitude is not None:
            if not (-90 <= location.latitude <= 90):
                issues.append(f"Invalid latitude: {location.latitude}")
        
        if location.longitude is not None:
            if not (-180 <= location.longitude <= 180):
                issues.append(f"Invalid longitude: {location.longitude}")
        
        # Validate state
        if location.state and location.state not in self.state_abbreviations:
            issues.append(f"Invalid state abbreviation: {location.state}")
        
        # Validate ZIP code format
        if location.zip_code:
            if not re.match(r'^\d{5}(?:-\d{4})?$', location.zip_code):
                issues.append(f"Invalid ZIP code format: {location.zip_code}")
        
        return issues
    
    def get_location_stats(self) -> Dict:
        """Get statistics about cached locations."""
        if not self.location_cache:
            return {}
        
        total_locations = len(self.location_cache)
        geocoded_count = sum(1 for loc in self.location_cache.values() 
                           if loc.latitude and loc.longitude)
        
        accuracy_counts = {}
        for location in self.location_cache.values():
            if location.accuracy:
                accuracy_counts[location.accuracy] = accuracy_counts.get(location.accuracy, 0) + 1
        
        return {
            'total_cached_locations': total_locations,
            'geocoded_locations': geocoded_count,
            'geocoded_percentage': (geocoded_count / total_locations) * 100 if total_locations > 0 else 0,
            'accuracy_breakdown': accuracy_counts,
            'jurisdictions_loaded': len(self.jurisdictions)
        }


def main():
    """Main function for testing the geo locator."""
    geo_locator = GeoLocator()
    
    # Test address parsing
    test_addresses = [
        "123 Main St, Springfield, IL 62701",
        "Springfield, IL",
        "Chicago, IL 60601",
        "New York, NY",
        "Invalid address format"
    ]
    
    print("Testing address parsing:")
    for address in test_addresses:
        location = geo_locator.parse_address(address)
        print(f"'{address}' -> {location.to_dict()}")
        
        # Validate the location
        issues = geo_locator.validate_location(location)
        if issues:
            print(f"  Issues: {', '.join(issues)}")
        
        # Try geocoding
        geocoded = geo_locator.geocode_location(location)
        if geocoded.latitude and geocoded.longitude:
            print(f"  Geocoded: {geocoded.latitude}, {geocoded.longitude}")
        
        print()
    
    # Test distance calculation
    loc1 = Location(city="Chicago", state="IL", latitude=41.8781, longitude=-87.6298)
    loc2 = Location(city="Springfield", state="IL", latitude=39.7817, longitude=-89.6501)
    
    distance = geo_locator.calculate_distance(loc1, loc2)
    if distance:
        print(f"Distance between Chicago and Springfield: {distance:.1f} miles")
    
    # Print stats
    stats = geo_locator.get_location_stats()
    print(f"Geo locator statistics: {stats}")


if __name__ == "__main__":
    main()