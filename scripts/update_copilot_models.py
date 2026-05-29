#!/usr/bin/env python3
"""
Update GitHub Copilot models JSON from official documentation.

This script fetches the latest supported AI models from GitHub's documentation
and updates a local JSON file with the current model information.
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: Required dependencies not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Configuration
DOCS_URL = "https://docs.github.com/en/copilot/reference/ai-models/supported-models"
SOURCE_URL = "https://docs.github.com/en/enterprise-cloud@latest/copilot/reference/ai-models/supported-models#supported-ai-models-in-copilot"
OUTPUT_FILE = "copilot_models.json"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Returns:
        Configured requests.Session with retry adapter
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def fetch_docs_content(session: requests.Session) -> str:
    """
    Fetch the documentation page content from GitHub Docs.
    
    Args:
        session: Configured requests session
        
    Returns:
        Raw HTML content from the docs page
        
    Raises:
        requests.RequestException: If fetching fails after retries
    """
    logger.info(f"Fetching documentation from: {DOCS_URL}")
    
    try:
        response = session.get(
            DOCS_URL,
            timeout=REQUEST_TIMEOUT,
            headers={
                'User-Agent': 'GitHub-Copilot-Models-Updater/1.0',
                'Accept': 'text/html,application/xhtml+xml'
            }
        )
        response.raise_for_status()
        
        content = response.text
        logger.info(f"Successfully fetched {len(content)} bytes of content")
        return content
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch documentation: {e}")
        raise


def parse_html_table(soup: BeautifulSoup, section_id: str) -> List[Dict[str, str]]:
    """
    Parse an HTML table following a specific section heading.
    
    Args:
        soup: BeautifulSoup object of the page
        section_id: ID or text of the section heading
        
    Returns:
        List of dictionaries representing table rows
    """
    def normalize_header(text: str) -> str:
        """Normalize header text to proper snake_case."""
        # Convert to lowercase and replace spaces with underscores
        normalized = text.lower().replace(' ', '_')
        # Fix compound words: "forpaid" -> "for_paid", "forcopilot" -> "for_copilot"
        # These specific patterns appear in the table headers
        normalized = re.sub(r'for(paid|copilot|free)', r'for_\1', normalized)
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove markdown formatting like ** 
        normalized = normalized.replace('*', '')
        return normalized
    
    # Find the section heading
    section = soup.find('h2', id=section_id)
    if not section:
        # Try finding by text content
        section = soup.find('h2', string=re.compile(section_id, re.IGNORECASE))
    
    if not section:
        logger.warning(f"Section '{section_id}' not found")
        return []
    
    # Find the next table after this heading
    table = section.find_next('table')
    if not table:
        logger.warning(f"No table found after section '{section_id}'")
        return []
    
    # Extract headers
    headers = []
    header_row = table.find('thead')
    if header_row:
        for th in header_row.find_all('th'):
            header_text = normalize_header(th.get_text(strip=True))
            headers.append(header_text)
    
    if not headers:
        # Try getting headers from first row if no thead
        first_row = table.find('tr')
        if first_row:
            for th in first_row.find_all(['th', 'td']):
                header_text = normalize_header(th.get_text(strip=True))
                headers.append(header_text)
    
    logger.debug(f"Found table headers: {headers}")
    
    # Extract data rows
    table_rows = []
    tbody = table.find('tbody')
    if tbody:
        # If tbody exists, get all rows from tbody
        rows = tbody.find_all('tr')
    else:
        # If no tbody, skip the first row (which we used for headers)
        all_rows = table.find_all('tr')
        rows = all_rows[1:] if len(all_rows) > 1 else []
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
            
        cell_values = [cell.get_text(strip=True) for cell in cells]
        
        # Skip empty rows
        if not any(cell_values):
            continue
        
        # Create dictionary from headers and values, omitting empty strings
        row_dict = {}
        for i, header in enumerate(headers):
            if i < len(cell_values):
                value = cell_values[i]
                # Only add non-empty values
                if value:
                    row_dict[header] = value
        
        table_rows.append(row_dict)
    
    logger.info(f"Parsed {len(table_rows)} rows from section '{section_id}'")
    return table_rows


def extract_models_data(content: str) -> List[Dict[str, Any]]:
    """
    Extract model information from the documentation HTML content.
    
    Args:
        content: HTML content from docs page
        
    Returns:
        List of model dictionaries with normalized structure
    """
    soup = BeautifulSoup(content, 'html.parser')
    models = []
    
    # Parse main models table
    main_models = parse_html_table(soup, 'supported-ai-models-in-copilot')
    
    # Parse retirement history
    retired_models = parse_html_table(soup, 'model-retirement-history')
    retired_dict = {m.get('model_name', ''): m for m in retired_models}
    
    # Parse per-client availability
    client_models = parse_html_table(soup, 'supported-ai-models-per-client')
    client_dict = {m.get('model', ''): m for m in client_models}
    
    # Parse per-plan availability
    plan_models = parse_html_table(soup, 'supported-ai-models-per-copilot-plan')
    plan_dict = {m.get('available_models_in_chat', ''): m for m in plan_models}
    
    # Parse model multipliers
    multiplier_models = parse_html_table(soup, 'model-multipliers')
    multiplier_dict = {m.get('model', ''): m for m in multiplier_models}
    
    # Combine data from all tables
    for model_data in main_models:
        model_name = model_data.get('model_name', '')
        
        if not model_name:
            continue
        
        # Build availability object, only including non-empty values
        availability = {}
        if model_data.get('release_status'):
            availability['release_status'] = model_data['release_status']
        if model_data.get('agent_mode'):
            availability['agent_mode'] = model_data['agent_mode']
        if model_data.get('ask_mode'):
            availability['ask_mode'] = model_data['ask_mode']
        if model_data.get('edit_mode'):
            availability['edit_mode'] = model_data['edit_mode']
        
        model = {
            'name': model_name
        }
        
        # Only add provider if present
        if model_data.get('provider'):
            model['provider'] = model_data['provider']
        
        # Only add availability if it has content
        if availability:
            model['availability'] = availability
        
        # Add client availability (omitting empty values)
        if model_name in client_dict:
            clients = {
                k: v for k, v in client_dict[model_name].items()
                if k != 'model' and v  # Omit empty values
            }
            if clients:
                if 'availability' not in model:
                    model['availability'] = {}
                model['availability']['clients'] = clients
        
        # Add plan availability (omitting empty values)
        if model_name in plan_dict:
            plans = {
                k: v for k, v in plan_dict[model_name].items()
                if k != 'available_models_in_chat' and v  # Omit empty values
            }
            if plans:
                if 'availability' not in model:
                    model['availability'] = {}
                model['availability']['plans'] = plans
        
        # Add multiplier info (omitting empty values)
        if model_name in multiplier_dict:
            multipliers = {
                k: v for k, v in multiplier_dict[model_name].items()
                if k != 'model' and v  # Omit empty values
            }
            if multipliers:
                model['multiplier'] = multipliers
        
        # Add retirement info if applicable
        if model_name in retired_dict:
            model['retirement'] = retired_dict[model_name]
            # Only add notes field when there are notes
            retirement_date = retired_dict[model_name].get('retirement_date', 'N/A')
            model['notes'] = [f"Retired: {retirement_date}"]
        
        # Set last_seen timestamp
        model['last_seen'] = datetime.now(timezone.utc).isoformat()
        
        models.append(model)
    
    # Also add retired models that aren't in the main table
    for retired_name, retired_data in retired_dict.items():
        if not any(m['name'] == retired_name for m in models):
            model = {
                'name': retired_name,
                'availability': {
                    'release_status': 'retired'
                },
                'retirement': retired_data,
                'notes': [f"Retired: {retired_data.get('retirement_date', 'N/A')}"],
                'last_seen': datetime.now(timezone.utc).isoformat()
            }
            models.append(model)
    
    logger.info(f"Extracted {len(models)} total models")
    return models


def generate_json_output(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate the final JSON structure with metadata.
    
    Args:
        models: List of model dictionaries
        
    Returns:
        Complete JSON structure with metadata
    """
    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_url': SOURCE_URL,
        'models': models
    }
    
    return output


def validate_json_schema(data: Dict[str, Any]) -> bool:
    """
    Validate that the JSON data conforms to the expected schema.
    
    Args:
        data: JSON data to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['generated_at', 'source_url', 'models']
    
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required key: {key}")
            return False
    
    if not isinstance(data['models'], list):
        logger.error("'models' must be a list")
        return False
    
    for model in data['models']:
        if 'name' not in model:
            logger.error(f"Model missing required 'name' field: {model}")
            return False
    
    logger.info("JSON schema validation passed")
    return True


def compute_content_hash(data: Dict[str, Any]) -> str:
    """
    Compute a hash of the model data (excluding timestamp).
    
    Args:
        data: JSON data
        
    Returns:
        SHA256 hash of the models content
    """
    # Create a copy without the timestamp for comparison
    models_only = data.get('models', [])
    
    # Remove last_seen timestamps for comparison (use deep copy for nested dicts)
    models_for_hash = []
    for model in models_only:
        model_copy = copy.deepcopy(model)
        model_copy.pop('last_seen', None)
        models_for_hash.append(model_copy)
    
    content_str = json.dumps(models_for_hash, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def has_content_changed(new_data: Dict[str, Any], output_path: str) -> bool:
    """
    Check if the content has changed compared to the existing file.
    
    Args:
        new_data: New JSON data
        output_path: Path to existing JSON file
        
    Returns:
        True if content changed, False otherwise
    """
    if not os.path.exists(output_path):
        logger.info("Output file doesn't exist, will create new file")
        return True
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        old_hash = compute_content_hash(existing_data)
        new_hash = compute_content_hash(new_data)
        
        if old_hash == new_hash:
            logger.info("Content unchanged (hash match)")
            return False
        else:
            logger.info("Content changed (hash mismatch)")
            logger.debug(f"Old hash: {old_hash}")
            logger.debug(f"New hash: {new_hash}")
            return True
            
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read existing file: {e}")
        return True


def write_json_file(data: Dict[str, Any], output_path: str) -> None:
    """
    Write JSON data to file with proper formatting.
    
    Args:
        data: JSON data to write
        output_path: Output file path
    """
    logger.info(f"Writing JSON to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline
    
    logger.info(f"Successfully wrote {len(data['models'])} models to {output_path}")


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description='Update GitHub Copilot models JSON from documentation'
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_FILE,
        help=f'Output JSON file path (default: {OUTPUT_FILE})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if content unchanged'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch and parse but do not write file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Create session with retry logic
        session = create_session()
        
        # Fetch documentation content
        content = fetch_docs_content(session)
        
        # Extract model data
        models = extract_models_data(content)
        
        if not models:
            logger.error("No models extracted from documentation")
            return 1
        
        # Generate JSON output
        json_data = generate_json_output(models)
        
        # Validate schema
        if not validate_json_schema(json_data):
            logger.error("JSON schema validation failed")
            return 1
        
        # Check if content changed
        if not args.force and not args.dry_run:
            if not has_content_changed(json_data, args.output):
                logger.info("No changes detected, skipping file write")
                return 0
        
        # Write output
        if args.dry_run:
            logger.info("Dry run mode - would write:")
            print(json.dumps(json_data, indent=2))
        else:
            write_json_file(json_data, args.output)
        
        logger.info("Update completed successfully")
        return 0
        
    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
