import requests
import time
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import json
from datetime import datetime, timedelta

from config.settings import Settings

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    pass

class APIHelper:
    """Helper class for making robust API calls with rate limiting and error handling."""
    
    def __init__(self, rate_limit_config: Optional[Dict] = None):
        self.rate_limit_config = rate_limit_config or Settings.API_RATE_LIMITS['rapidapi']
        self.last_request_time = {}
        self.request_counts = {}
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, RateLimitError))
    )
    def make_request(self, 
                    url: str, 
                    headers: Dict[str, str], 
                    params: Optional[Dict] = None,
                    timeout: int = 30) -> Dict[str, Any]:
        """
        Make a robust HTTP request with retry logic and rate limiting.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing the API response
            
        Raises:
            APIError: If request fails after all retries
            RateLimitError: If rate limit is exceeded
        """
        # Rate limiting
        self._check_rate_limit(url)
        
        try:
            logger.debug(f"Making request to: {url}")
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                raise RateLimitError(f"Rate limited for {retry_after} seconds")
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Track successful request
            self._track_request(url)
            
            # Parse JSON response
            data = response.json()
            
            # Log API usage if available
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                logger.info(f"API requests remaining: {remaining}")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for URL: {url}")
            raise APIError(f"Request timeout: {url}")
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for URL: {url}")
            raise APIError(f"Connection error: {url}")
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for URL: {url}")
            raise APIError(f"HTTP {e.response.status_code}: {url}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from URL: {url}")
            raise APIError(f"Invalid JSON response: {url}")
    
    def _check_rate_limit(self, url: str):
        """Check if we're within rate limits for this URL."""
        domain = self._extract_domain(url)
        current_time = time.time()
        
        # Initialize tracking for new domains
        if domain not in self.last_request_time:
            self.last_request_time[domain] = current_time
            self.request_counts[domain] = 0
            return
        
        # Check if enough time has passed since last request
        time_since_last = current_time - self.last_request_time[domain]
        min_interval = 60 / self.rate_limit_config['requests_per_minute']
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    def _track_request(self, url: str):
        """Track request for rate limiting purposes."""
        domain = self._extract_domain(url)
        self.last_request_time[domain] = time.time()
        self.request_counts[domain] = self.request_counts.get(domain, 0) + 1
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting tracking."""
        from urllib.parse import urlparse
        return urlparse(url).netloc

def get_all_pages(path: str, 
                 host: str, 
                 headers: Dict[str, str], 
                 max_pages: int = 10,
                 **params) -> List[Dict[str, Any]]:
    """
    Fetch all pages from a paginated API endpoint.
    
    Args:
        path: API endpoint path
        host: API host
        headers: Request headers
        max_pages: Maximum number of pages to fetch
        **params: Query parameters
        
    Returns:
        List of all items from all pages
    """
    api_helper = APIHelper()
    all_results = []
    page = 1
    
    while page <= max_pages:
        url = f"https://{host}{path}"
        current_params = {**params, 'page': page}
        
        try:
            data = api_helper.make_request(url, headers, current_params)
            
            # Handle different response structures
            items = []
            if 'response' in data:
                items = data['response']
            elif 'results' in data:
                items = data['results']
            elif 'data' in data:
                items = data['data']
            else:
                items = data if isinstance(data, list) else []
            
            if not items:
                logger.info(f"No more items on page {page}")
                break
            
            all_results.extend(items)
            
            # Check pagination info
            paging = data.get('paging', {})
            current_page = paging.get('current', page)
            total_pages = paging.get('total', page)
            
            logger.info(f"Fetched page {current_page}/{total_pages} with {len(items)} items")
            
            if current_page >= total_pages:
                break
                
            page += 1
            
        except APIError as e:
            logger.error(f"Error fetching page {page}: {e}")
            break
    
    logger.info(f"Fetched total of {len(all_results)} items across {page-1} pages")
    return all_results

async def make_async_request(session: aiohttp.ClientSession,
                           url: str,
                           headers: Dict[str, str],
                           params: Optional[Dict] = None,
                           timeout: int = 30) -> Dict[str, Any]:
    """
    Make an asynchronous HTTP request.
    
    Args:
        session: aiohttp client session
        url: API endpoint URL
        headers: Request headers
        params: Query parameters
        timeout: Request timeout
        
    Returns:
        Dict containing the API response
    """
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with session.get(url, headers=headers, params=params, timeout=timeout_obj) as response:
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                raise RateLimitError(f"Rate limited for {retry_after} seconds")
            
            response.raise_for_status()
            return await response.json()
            
    except asyncio.TimeoutError:
        logger.error(f"Async request timeout for URL: {url}")
        raise APIError(f"Async request timeout: {url}")
    except aiohttp.ClientError as e:
        logger.error(f"Async request error for URL: {url} - {e}")
        raise APIError(f"Async request error: {url}")

async def fetch_multiple_endpoints(requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fetch multiple API endpoints concurrently.
    
    Args:
        requests_list: List of request dictionaries with 'url', 'headers', 'params'
        
    Returns:
        List of response dictionaries
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for req in requests_list:
            task = make_async_request(
                session=session,
                url=req['url'],
                headers=req['headers'],
                params=req.get('params'),
                timeout=req.get('timeout', 30)
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                    processed_results.append({'error': str(result)})
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in concurrent requests: {e}")
            return [{'error': str(e)} for _ in requests_list]

def create_api_sports_headers() -> Dict[str, str]:
    """
    Create headers for direct API-Sports requests.
    
    Returns:
        Dict of headers
    """
    return {
        "x-apisports-key": Settings.API_SPORTS_KEY,
        "Accept": "application/json"
    }

def create_odds_api_headers() -> Dict[str, str]:
    """
    Create headers for The Odds API requests.
    
    Returns:
        Dict of headers
    """
    return {
        "Accept": "application/json"
    }

def validate_api_response(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate that an API response contains required fields.
    
    Args:
        data: API response data
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present
    """
    if not isinstance(data, dict):
        return False
    
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return False
    
    return True

def extract_error_message(response_data: Dict[str, Any]) -> str:
    """
    Extract error message from API response.
    
    Args:
        response_data: API response data
        
    Returns:
        Error message string
    """
    # Common error field names
    error_fields = ['error', 'message', 'errors', 'error_message', 'description']
    
    for field in error_fields:
        if field in response_data:
            error_value = response_data[field]
            if isinstance(error_value, str):
                return error_value
            elif isinstance(error_value, list) and error_value:
                return str(error_value[0])
            elif isinstance(error_value, dict):
                return json.dumps(error_value)
    
    return "Unknown API error"

def cache_key_generator(sport: str, endpoint: str, **params) -> str:
    """
    Generate cache key for API responses.
    
    Args:
        sport: Sport name (nba, mlb, nfl)
        endpoint: API endpoint name
        **params: Query parameters
        
    Returns:
        Cache key string
    """
    param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()) if v is not None)
    timestamp = datetime.now().strftime("%Y%m%d_%H")  # Hour-based caching
    return f"{sport}_{endpoint}_{param_str}_{timestamp}"

def is_cache_valid(cache_time: datetime, max_age_seconds: int) -> bool:
    """
    Check if cached data is still valid.
    
    Args:
        cache_time: When data was cached
        max_age_seconds: Maximum age in seconds
        
    Returns:
        True if cache is still valid
    """
    age = (datetime.now() - cache_time).total_seconds()
    return age < max_age_seconds

def format_api_date(date_obj: Union[datetime, str]) -> str:
    """
    Format date for API requests (YYYY-MM-DD).
    
    Args:
        date_obj: Date object or string
        
    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        return date_obj
    elif isinstance(date_obj, datetime):
        return date_obj.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Invalid date type: {type(date_obj)}")

def get_current_season(sport: str) -> int:
    """
    Get current season year for a sport.
    
    Args:
        sport: Sport name (nba, mlb, nfl)
        
    Returns:
        Season year
    """
    now = datetime.now()
    sport_config = Settings.SPORT_CONFIGS.get(sport.lower(), {})
    
    season_start_month = sport_config.get('season_start_month', 1)
    
    # If current month is before season start, we're in previous year's season
    if now.month < season_start_month:
        return now.year - 1
    else:
        return now.year
