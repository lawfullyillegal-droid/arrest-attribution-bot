# Arrest Attribution Bot

A comprehensive system for collecting, processing, and attributing arrest data from various sources. This bot helps law enforcement agencies and oversight organizations track arrest patterns, officer activities, and departmental statistics while maintaining data integrity and security.

## Features

- **Multi-source Data Collection**: Supports various data sources including jail rosters, booking logs, and online databases
- **Intelligent Attribution**: Matches officers and departments across different data sources
- **Data Normalization**: Standardizes arrest event data from various formats
- **Timestamp Verification**: Validates and verifies event timestamps for accuracy
- **Secure Logging**: Hash-based logging system for audit trails and data integrity
- **Geolocation Services**: Geocoding and jurisdiction mapping capabilities
- **Comprehensive Reporting**: Generates affidavit logs and analytical dashboards

## Architecture

```
arrest-attribution-bot/
├── data_sources/           # Data collection and processing
│   ├── jail_rosters_scraper.py    # Web scraping for jail rosters
│   ├── booking_logs_parser.py     # Parse various log file formats
│   └── arrest_event_normalizer.py # Normalize data to standard format
├── attribution_engine/     # Core attribution logic
│   ├── officer_matcher.py         # Match officers across sources
│   ├── department_mapper.py       # Standardize department names
│   └── timestamp_verifier.py      # Validate event timestamps
├── output/                 # Generated reports and logs
│   ├── affidavit_logs/           # Legal document exports
│   └── dashboards/               # Analytics and summaries
├── utils/                  # Utility modules
│   ├── hash_logger.py            # Secure audit logging
│   └── geo_locator.py            # Location services
├── main.py                # Main application entry point
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lawfullyillegal-droid/arrest-attribution-bot.git
cd arrest-attribution-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Test Mode)

Run the bot with sample data to test functionality:

```bash
python main.py --test
```

This will:
- Create sample arrest data
- Process the data through the full pipeline
- Generate reports in the `output/` directory
- Create audit logs in the `logs/` directory

### Production Usage

1. Create a data sources configuration file (e.g., `sources.json`):

```json
[
  {
    "type": "file",
    "path": "/path/to/booking_logs.csv",
    "name": "county_jail_bookings"
  },
  {
    "type": "url",
    "urls": ["https://example.com/jail-roster"],
    "name": "live_jail_roster"
  }
]
```

2. Run the attribution pipeline:

```bash
python main.py --sources sources.json
```

## Core Components

### Data Sources

#### Jail Rosters Scraper
- Scrapes live jail roster data from law enforcement websites
- Handles rate limiting and respectful scraping practices
- Supports multiple URL sources

#### Booking Logs Parser
- Parses various file formats (JSON, CSV, XML, TXT)
- Normalizes field names and data structures
- Handles multiple encoding formats

#### Arrest Event Normalizer
- Converts raw data into standardized `NormalizedArrestEvent` objects
- Generates unique event IDs
- Categorizes charges and validates data integrity

### Attribution Engine

#### Officer Matcher
- Fuzzy matching of officer names across data sources
- Maintains officer profiles with aliases and badge numbers
- Confidence scoring for match quality

#### Department Mapper
- Standardizes department/agency names
- Maps abbreviations to full names
- Handles jurisdiction type classification

#### Timestamp Verifier
- Validates event timestamp sequences
- Checks for logical consistency (arrest before booking, etc.)
- Supports multiple date/time formats

### Utilities

#### Hash Logger
- Secure audit trail using cryptographic hashes
- PII access logging
- Data modification tracking
- Tamper-evident log entries

#### Geo Locator
- Address parsing and geocoding
- Jurisdiction boundary detection
- Distance calculations
- Location validation

## Configuration

The system supports various configuration options:

### Data Source Types

- **File Sources**: Local files in JSON, CSV, XML, or TXT format
- **URL Sources**: Web-based jail rosters and booking systems
- **Database Sources**: (Future enhancement)

### Output Formats

- **Affidavit Logs**: JSON format suitable for legal documentation
- **Dashboards**: Summary statistics and analytics
- **Audit Reports**: Complete hash-based audit trails

## Security Features

- **Hash-based Logging**: All sensitive data access is logged with cryptographic hashes
- **PII Protection**: Personally identifiable information is hashed for security
- **Audit Trails**: Complete tracking of data modifications and access
- **Secure IDs**: Deterministic ID generation without exposing sensitive data

## Legal and Ethical Considerations

This tool is designed to promote transparency and accountability in law enforcement while respecting privacy and legal requirements:

- **Public Records Only**: Only processes publicly available arrest data
- **Privacy Protection**: Implements secure hashing for sensitive information
- **Audit Compliance**: Maintains comprehensive audit trails
- **Responsible Use**: Intended for legitimate oversight and research purposes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is released under the MIT License. See LICENSE file for details.

## Disclaimer

This software is provided for educational and research purposes. Users are responsible for compliance with applicable laws and regulations regarding data collection, processing, and privacy. The authors assume no liability for misuse of this software.