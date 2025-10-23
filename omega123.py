import yaml
import os
from typing import Dict, Any

class Config:
    """Configuration loader for SF Assist application"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file in the current directory."
            )
        
        try:
            # Use UTF-8 encoding explicitly to avoid encoding issues
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required keys
            required_keys = ['api', 'mcp', 'app']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration section: {key}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Encoding error reading config file: {e}\n"
                f"Please ensure config.yaml is saved with UTF-8 encoding."
            )
        
        # Override with environment variables if they exist
        if os.getenv('SF_ASSIST_API_KEY'):
            config['api']['api_key'] = os.getenv('SF_ASSIST_API_KEY')
        if os.getenv('SF_ASSIST_API_URL'):
            config['api']['url'] = os.getenv('SF_ASSIST_API_URL')
        if os.getenv('MCP_SERVER_URL'):
            config['mcp']['default_url'] = os.getenv('MCP_SERVER_URL')
        
        return config
    
    @property
    def api_url(self) -> str:
        return self._config['api']['url']
    
    @property
    def api_key(self) -> str:
        return self._config['api']['api_key']
    
    @property
    def app_id(self) -> str:
        return self._config['api']['app_id']
    
    @property
    def aplctn_cd(self) -> str:
        return self._config['api']['aplctn_cd']
    
    @property
    def model(self) -> str:
        return self._config['api']['model']
    
    @property
    def sys_msg(self) -> str:
        return self._config['api']['sys_msg']
    
    @property
    def verify_ssl(self) -> bool:
        return self._config['api']['verify_ssl']
    
    @property
    def mcp_default_url(self) -> str:
        return self._config['mcp']['default_url']
    
    @property
    def mcp_server_name(self) -> str:
        return self._config['mcp']['server_name']
    
    @property
    def mcp_transport(self) -> str:
        return self._config['mcp']['transport']
    
    @property
    def app_title(self) -> str:
        return self._config['app']['title']
    
    @property
    def app_icon(self) -> str:
        return self._config['app']['icon']
    
    @property
    def app_version(self) -> str:
        return self._config['app']['version']

# Global config instance
config = Config()
