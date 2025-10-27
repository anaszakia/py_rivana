"""
Configuration Manager untuk Hidrologi ML API
Supports: Local Development & Production VPS
"""
import os
import sys
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class that auto-detects environment"""
    
    def __init__(self):
        # Detect environment
        self.environment = os.getenv('ENVIRONMENT', self._detect_environment())
        
        # Load .env file
        self._load_env_file()
        
        # API Configuration
        self.API_HOST = self._clean_value(os.getenv('API_HOST', '127.0.0.1'))
        self.API_PORT = int(os.getenv('API_PORT', '8001'))
        self.DEBUG = self._clean_value(os.getenv('DEBUG', 'False')).lower() == 'true'
        
        # Paths
        self.RESULTS_DIR = self._get_path('RESULTS_DIR', 'results')
        self.TEMP_DIR = self._get_path('TEMP_DIR', 'temp')
        
        # Earth Engine
        self.EE_AUTHENTICATED = self._clean_value(os.getenv('EE_AUTHENTICATED', 'True')).lower() == 'true'
        
        # Security - Clean quotes from token values
        self.SECRET_KEY = self._clean_value(os.getenv('SECRET_KEY', '893d3a0b7779b5b08fca2b83527a7c388b8d2dab79afee728861866f10b4f5f1'))
        self.API_TOKEN = self._clean_value(os.getenv('API_TOKEN', 'a7f3e9d2b1c8f4a6e2d9b7c3f1a8e5d4'))  # Bearer Token for API authentication
        
        # Limits
        self.MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '2'))
        self.JOB_TIMEOUT = int(os.getenv('JOB_TIMEOUT', '1800'))
        
        # Logging
        self.LOG_LEVEL = self._clean_value(os.getenv('LOG_LEVEL', 'DEBUG' if self.DEBUG else 'INFO'))
        self.LOG_FILE = self._clean_value(os.getenv('LOG_FILE', 'logs/api.log'))
        
        # Performance
        self.ENABLE_CORS = self._clean_value(os.getenv('ENABLE_CORS', 'True')).lower() == 'true'
        self.ENABLE_CACHE = self._clean_value(os.getenv('ENABLE_CACHE', 'False')).lower() == 'true'
        self.CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))
        
        # Rate Limiting
        self.RATE_LIMIT_ENABLED = self._clean_value(os.getenv('RATE_LIMIT_ENABLED', 'False')).lower() == 'true'
        self.RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
        self.RATE_LIMIT_PERIOD = int(os.getenv('RATE_LIMIT_PERIOD', '60'))
        
        # Create directories if not exist
        self._ensure_directories()
    
    def _detect_environment(self) -> str:
        """Auto-detect environment based on OS and paths"""
        if sys.platform == 'win32':
            return 'local'
        elif os.path.exists('/home/hidrologi') or os.path.exists('/var/log/hidrologi'):
            return 'production'
        else:
            return 'local'
    
    def _load_env_file(self):
        """Load appropriate .env file based on environment - OPTIMIZED"""
        # ⚡ SKIP dotenv jika tidak ter-install (untuk kecepatan)
        # Cukup gunakan environment variables atau defaults
        try:
            from dotenv import load_dotenv
            
            # Try to load environment-specific file first
            env_file = f'.env.{self.environment}'
            if os.path.exists(env_file):
                load_dotenv(env_file, override=False)  # Don't override existing env vars
                # Removed print to reduce overhead
            elif os.path.exists('.env'):
                load_dotenv('.env', override=False)
                # Removed print to reduce overhead
        except ImportError:
            # Silently skip - use environment variables or defaults
            # No print to reduce overhead
            pass
    
    def _clean_value(self, value: str) -> str:
        """Remove quotes from environment variable values"""
        if not value:
            return value
        # Remove surrounding quotes (single or double)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value
    
    def _get_path(self, env_var: str, default: str) -> str:
        """Get path from environment, handle relative/absolute paths"""
        path = os.getenv(env_var, default)
        
        # If relative path, make it absolute based on project root
        if not os.path.isabs(path):
            project_root = Path(__file__).parent.parent
            path = str(project_root / path)
        
        return path
    
    def _ensure_directories(self):
        """Create required directories if they don't exist"""
        dirs_to_create = [
            self.RESULTS_DIR,
            self.TEMP_DIR,
            os.path.dirname(self.LOG_FILE) if '/' in self.LOG_FILE or '\\' in self.LOG_FILE else 'logs'
        ]
        
        for dir_path in dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not create directory {dir_path}: {e}")
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == 'production'
    
    def is_local(self) -> bool:
        """Check if running in local development"""
        return self.environment == 'local'
    
    def print_config(self):
        """Print current configuration (for debugging)"""
        print("\n" + "="*80)
        print(f"🔧 CONFIGURATION - {self.environment.upper()} MODE")
        print("="*80)
        print(f"Environment:     {self.environment}")
        print(f"Debug Mode:      {self.DEBUG}")
        print(f"API Host:        {self.API_HOST}:{self.API_PORT}")
        print(f"API Token:       {'*' * 20}...{self.API_TOKEN[-4:] if len(self.API_TOKEN) > 4 else '****'}")  # Hide token
        print(f"Results Dir:     {self.RESULTS_DIR}")
        print(f"Temp Dir:        {self.TEMP_DIR}")
        print(f"Log Level:       {self.LOG_LEVEL}")
        print(f"Log File:        {self.LOG_FILE}")
        print(f"Max Jobs:        {self.MAX_CONCURRENT_JOBS}")
        print(f"Job Timeout:     {self.JOB_TIMEOUT}s")
        print(f"CORS Enabled:    {self.ENABLE_CORS}")
        print(f"Cache Enabled:   {self.ENABLE_CACHE}")
        print(f"Rate Limit:      {self.RATE_LIMIT_ENABLED}")
        print("="*80 + "\n")


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_config()
    
    # Test directory creation
    print("Testing directory creation...")
    print(f"Results dir exists: {os.path.exists(config.RESULTS_DIR)}")
    print(f"Temp dir exists: {os.path.exists(config.TEMP_DIR)}")
