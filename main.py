"""
Main Entry Point for Arrest Attribution Bot

This is the main application that coordinates all components of the arrest attribution system.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

# Import our modules
from data_sources.jail_rosters_scraper import JailRosterScraper
from data_sources.booking_logs_parser import BookingLogsParser
from data_sources.arrest_event_normalizer import ArrestEventNormalizer, NormalizedArrestEvent
from attribution_engine.officer_matcher import OfficerMatcher, Officer
from attribution_engine.department_mapper import DepartmentMapper, Department
from attribution_engine.timestamp_verifier import TimestampVerifier
from utils.hash_logger import HashLogger
from utils.geo_locator import GeoLocator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('arrest_attribution.log')
    ]
)

logger = logging.getLogger(__name__)


class ArrestAttributionBot:
    """Main application class that coordinates all components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the arrest attribution bot.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.scraper = JailRosterScraper()
        self.parser = BookingLogsParser()
        self.normalizer = ArrestEventNormalizer()
        self.officer_matcher = OfficerMatcher()
        self.department_mapper = DepartmentMapper()
        self.timestamp_verifier = TimestampVerifier()
        self.hash_logger = HashLogger(log_file='logs/hash_audit.log')
        self.geo_locator = GeoLocator()
        
        # Initialize output directories
        self._setup_output_directories()
        
        logger.info("Arrest Attribution Bot initialized")
    
    def _setup_output_directories(self):
        """Setup output directories."""
        output_dirs = [
            'output/affidavit_logs',
            'output/dashboards',
            'logs'
        ]
        
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_data_sources(self, sources: List[Dict]) -> List[NormalizedArrestEvent]:
        """
        Process multiple data sources and return normalized events.
        
        Args:
            sources: List of source configurations
            
        Returns:
            List of normalized arrest events
        """
        all_events = []
        
        for source_config in sources:
            source_type = source_config.get('type', 'unknown')
            source_path = source_config.get('path')
            source_urls = source_config.get('urls', [])
            
            logger.info(f"Processing source: {source_type}")
            
            try:
                if source_type == 'file':
                    events = self._process_file_source(source_path, source_config)
                elif source_type == 'url':
                    events = self._process_url_source(source_urls, source_config)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                all_events.extend(events)
                
            except Exception as e:
                logger.error(f"Failed to process source {source_type}: {e}")
                continue
        
        logger.info(f"Processed {len(all_events)} total events from {len(sources)} sources")
        return all_events
    
    def _process_file_source(self, file_path: str, config: Dict) -> List[NormalizedArrestEvent]:
        """Process file-based data source."""
        if not file_path or not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Parse the file
        raw_data = self.parser.parse_file(file_path)
        
        # Hash the raw data for audit
        self.hash_logger.log_data_hash(
            raw_data,
            'file_processed',
            metadata={
                'file_path': file_path,
                'record_count': len(raw_data)
            }
        )
        
        # Normalize the events
        source_name = config.get('name', f"file_{Path(file_path).stem}")
        normalized_events = self.normalizer.normalize_events(raw_data, source_name)
        
        return normalized_events
    
    def _process_url_source(self, urls: List[str], config: Dict) -> List[NormalizedArrestEvent]:
        """Process URL-based data source."""
        if not urls:
            return []
        
        # Configure scraper
        self.scraper.base_urls = urls
        
        # Scrape the data
        raw_data = self.scraper.scrape_all_sources()
        
        # Hash the raw data for audit
        self.hash_logger.log_data_hash(
            raw_data,
            'url_scraped',
            metadata={
                'urls': urls,
                'record_count': len(raw_data)
            }
        )
        
        # Normalize the events
        source_name = config.get('name', 'scraped_source')
        normalized_events = self.normalizer.normalize_events(raw_data, source_name)
        
        return normalized_events
    
    def attribute_events(self, events: List[NormalizedArrestEvent]) -> List[Dict[str, Any]]:
        """
        Perform attribution on normalized events.
        
        Args:
            events: List of normalized arrest events
            
        Returns:
            List of attributed event dictionaries
        """
        attributed_events = []
        
        for event in events:
            try:
                attributed_event = self._attribute_single_event(event)
                attributed_events.append(attributed_event)
            except Exception as e:
                logger.error(f"Failed to attribute event {event.event_id}: {e}")
                continue
        
        logger.info(f"Attributed {len(attributed_events)} events")
        return attributed_events
    
    def _attribute_single_event(self, event: NormalizedArrestEvent) -> Dict[str, Any]:
        """Attribute a single event."""
        attributed = {
            'event': event.to_dict(),
            'attribution': {
                'officer_match': None,
                'department_match': None,
                'timestamp_validation': None,
                'location_data': None
            },
            'confidence_scores': {},
            'attribution_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Officer attribution
        if event.arresting_officer:
            officer_data = {
                'name': event.arresting_officer,
                'department': event.department
            }
            
            officer_match = self.officer_matcher.match_officer(officer_data)
            if officer_match:
                attributed['attribution']['officer_match'] = {
                    'officer_id': officer_match.officer.officer_id,
                    'officer_name': officer_match.officer.name,
                    'badge_number': officer_match.officer.badge_number,
                    'department': officer_match.officer.department,
                    'match_criteria': officer_match.match_criteria
                }
                attributed['confidence_scores']['officer'] = officer_match.confidence_score
            else:
                # Create new officer if no match found
                new_officer = self.officer_matcher.create_officer_from_data(officer_data)
                self.officer_matcher.add_officer(new_officer)
                
                attributed['attribution']['officer_match'] = {
                    'officer_id': new_officer.officer_id,
                    'officer_name': new_officer.name,
                    'badge_number': new_officer.badge_number,
                    'department': new_officer.department,
                    'match_criteria': ['new_officer_created']
                }
                attributed['confidence_scores']['officer'] = 0.5  # Lower confidence for new officers
        
        # Department attribution
        if event.department:
            dept_match = self.department_mapper.map_department(event.department)
            if dept_match:
                attributed['attribution']['department_match'] = {
                    'department_id': dept_match.department.department_id,
                    'official_name': dept_match.department.official_name,
                    'common_name': dept_match.department.common_name,
                    'jurisdiction_type': dept_match.department.jurisdiction_type,
                    'match_type': dept_match.match_type
                }
                attributed['confidence_scores']['department'] = dept_match.confidence_score
            else:
                # Create new department if no match found
                new_dept = self.department_mapper.create_department_from_name(event.department, event.location)
                self.department_mapper.add_department(new_dept)
                
                attributed['attribution']['department_match'] = {
                    'department_id': new_dept.department_id,
                    'official_name': new_dept.official_name,
                    'common_name': new_dept.common_name,
                    'jurisdiction_type': new_dept.jurisdiction_type,
                    'match_type': 'new_department_created'
                }
                attributed['confidence_scores']['department'] = 0.5
        
        # Timestamp validation
        timestamp_sequence, timestamp_notes = self.timestamp_verifier.verify_event_sequence({
            'arrest_date': event.arrest_date,
            'booking_date': event.booking_date
        })
        
        attributed['attribution']['timestamp_validation'] = {
            'sequence_valid': timestamp_sequence.is_valid_sequence(),
            'notes': timestamp_notes,
            'arrest_time': timestamp_sequence.arrest_time.isoformat() if timestamp_sequence.arrest_time else None,
            'booking_time': timestamp_sequence.booking_time.isoformat() if timestamp_sequence.booking_time else None
        }
        
        attributed['confidence_scores']['timestamp'] = 1.0 if timestamp_sequence.is_valid_sequence() else 0.7
        
        # Location processing
        if event.location:
            location = self.geo_locator.parse_address(event.location)
            geocoded_location = self.geo_locator.geocode_location(location)
            
            attributed['attribution']['location_data'] = geocoded_location.to_dict()
            
            # Find jurisdiction
            jurisdiction = self.geo_locator.find_jurisdiction(geocoded_location)
            if jurisdiction:
                attributed['attribution']['location_data']['jurisdiction'] = {
                    'jurisdiction_id': jurisdiction.jurisdiction_id,
                    'name': jurisdiction.name,
                    'type': jurisdiction.jurisdiction_type
                }
            
            attributed['confidence_scores']['location'] = 0.9 if geocoded_location.is_complete() else 0.5
        
        # Calculate overall confidence
        confidence_values = list(attributed['confidence_scores'].values())
        attributed['overall_confidence'] = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        # Log the attribution for audit
        self.hash_logger.log_data_hash(
            attributed,
            'event_attributed',
            metadata={
                'event_id': event.event_id,
                'overall_confidence': attributed['overall_confidence']
            }
        )
        
        return attributed
    
    def generate_reports(self, attributed_events: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate reports from attributed events.
        
        Args:
            attributed_events: List of attributed events
            
        Returns:
            Dictionary with report file paths
        """
        reports = {}
        
        # Generate affidavit logs
        affidavit_path = self._generate_affidavit_logs(attributed_events)
        reports['affidavit_logs'] = affidavit_path
        
        # Generate summary dashboard
        dashboard_path = self._generate_dashboard(attributed_events)
        reports['dashboard'] = dashboard_path
        
        # Generate audit report
        audit_path = self._generate_audit_report()
        reports['audit_report'] = audit_path
        
        logger.info(f"Generated {len(reports)} reports")
        return reports
    
    def _generate_affidavit_logs(self, attributed_events: List[Dict[str, Any]]) -> str:
        """Generate affidavit logs."""
        import json
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"output/affidavit_logs/attributed_events_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(attributed_events, f, indent=2, default=str)
        
        logger.info(f"Generated affidavit logs: {filename}")
        return filename
    
    def _generate_dashboard(self, attributed_events: List[Dict[str, Any]]) -> str:
        """Generate dashboard summary."""
        import json
        
        # Calculate statistics
        total_events = len(attributed_events)
        high_confidence = sum(1 for e in attributed_events if e.get('overall_confidence', 0) > 0.8)
        
        officer_stats = self.officer_matcher.get_officer_stats()
        department_stats = self.department_mapper.get_department_stats()
        location_stats = self.geo_locator.get_location_stats()
        hash_stats = self.hash_logger.get_log_stats()
        
        dashboard_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_events_processed': total_events,
                'high_confidence_events': high_confidence,
                'high_confidence_percentage': (high_confidence / total_events) * 100 if total_events > 0 else 0
            },
            'officer_statistics': officer_stats,
            'department_statistics': department_stats,
            'location_statistics': location_stats,
            'audit_statistics': hash_stats
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"output/dashboards/summary_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Generated dashboard: {filename}")
        return filename
    
    def _generate_audit_report(self) -> str:
        """Generate audit report."""
        audit_logs = self.hash_logger.export_logs()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"logs/audit_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            import json
            json.dump(audit_logs, f, indent=2, default=str)
        
        logger.info(f"Generated audit report: {filename}")
        return filename
    
    def run_full_pipeline(self, sources: List[Dict]) -> Dict[str, Any]:
        """
        Run the complete attribution pipeline.
        
        Args:
            sources: List of data source configurations
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting full attribution pipeline")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Step 1: Process data sources
            normalized_events = self.process_data_sources(sources)
            
            # Step 2: Perform attribution
            attributed_events = self.attribute_events(normalized_events)
            
            # Step 3: Generate reports
            reports = self.generate_reports(attributed_events)
            
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'events_processed': len(normalized_events),
                'events_attributed': len(attributed_events),
                'reports_generated': reports
            }
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now(timezone.utc).isoformat()
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Arrest Attribution Bot')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--sources', help='Data sources configuration file')
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample data')
    
    args = parser.parse_args()
    
    # Initialize the bot
    bot = ArrestAttributionBot()
    
    if args.test:
        # Run with test data
        logger.info("Running in test mode")
        
        test_sources = [
            {
                'type': 'file',
                'path': 'sample_data.json',  # This would need to be created
                'name': 'test_file_source'
            }
        ]
        
        # Create sample data file if it doesn't exist
        sample_data_path = Path('sample_data.json')
        if not sample_data_path.exists():
            import json
            sample_data = [
                {
                    'name': 'John Doe',
                    'charge': 'Public Intoxication',
                    'arrest_date': '2024-01-15 14:30:00',
                    'officer': 'Officer Smith',
                    'department': 'City PD',
                    'location': 'Springfield, IL'
                },
                {
                    'name': 'Jane Smith',
                    'charge': 'Traffic Violation',
                    'arrest_date': '2024-01-15 16:45:00',
                    'officer': 'Officer Johnson',
                    'department': 'Highway Patrol',
                    'location': 'Chicago, IL'
                }
            ]
            
            with open(sample_data_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info("Created sample data file")
        
        results = bot.run_full_pipeline(test_sources)
        print(f"Test run results: {results}")
    
    else:
        # Run with provided configuration
        if args.sources:
            import json
            with open(args.sources, 'r') as f:
                sources = json.load(f)
            
            results = bot.run_full_pipeline(sources)
            print(f"Pipeline results: {results}")
        else:
            print("Please provide a sources configuration file or use --test mode")
            print("Example: python main.py --test")


if __name__ == "__main__":
    main()