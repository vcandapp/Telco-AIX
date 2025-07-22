"""
OpenShift AI Hosted Model Serving - Telco AIX - SME Chat UI
Author: Fatih E. NAR
"""

import gradio as gr
import requests
import json
from typing import List, Tuple, Optional, Dict, Any, Generator
from datetime import datetime, timedelta
import io
import base64
from dataclasses import dataclass
import urllib3
import traceback
import time
import threading
from urllib.parse import urljoin
import os
import uuid
import pickle
from pathlib import Path
from collections import deque, defaultdict
import re

# Plotting imports for metrics visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotly not available. Metrics will be displayed as text.")

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
@dataclass
class Config:
    """Enhanced configuration for the chat application"""
    api_endpoint: str = "https://qwen3-32b-vllm-latest-tme-aix.apps.sandbox01.narlabs.io"
    model_name: str = "qwen3-32b-vllm-latest"
    default_temperature: float = 0.3
    default_max_tokens: int = 4192
    admin_username: str = "admin"
    admin_password: str = "minad"
    verify_ssl: bool = False
    
    # Timeout settings
    connect_timeout: int = 45
    read_timeout: int = 240  # 4 minutes for non-streaming
    streaming_timeout: int = 600  # 10 minutes for streaming
    
    # Context management
    auto_stream_threshold: int = 4000  # Auto-enable streaming for large contexts
    max_file_chars: int = 3500  # Limit file content size
    max_retry_attempts: int = 5

# Load system prompts from external file
def load_system_prompts():
    """Load system prompts from external JSON file"""
    prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
    
    # Default compact prompts as fallback
    default_prompts = {
        "Default Assistant": "You are a helpful AI assistant. Provide direct, clear responses without showing your reasoning process.",
        "Technical Expert": "You are a technical expert in software engineering, cloud architecture, and AI/ML. Provide detailed, accurate responses directly without showing internal reasoning.",
        "Code Assistant": "You are an expert programmer. Write clean, well-documented code with explanations and best practices. Give direct answers without showing thinking process.",
        "Data Analyst": "You are a data analyst expert. Help analyze data, create insights, and explain statistical concepts clearly. Provide direct responses.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling, creative content, and engaging narratives. Give direct creative responses.",
        "Network Expert": "You are a network architect expert. Provide network design, troubleshooting, and optimization guidance.",
        "Telco Expert": "You are a telecommunications expert with expertise in 5G/4G/3G, RAN, Core networks, and telco standards. Provide technical responses with vendor comparisons.",
        "Custom": ""
    }
    
    try:
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                loaded_prompts = json.load(f)
                print(f"✅ Loaded {len(loaded_prompts)} system prompts from {prompts_file}")
                return loaded_prompts
        else:
            print(f"⚠️ System prompts file not found at {prompts_file}, using defaults")
            return default_prompts
    except Exception as e:
        print(f"❌ Error loading system prompts: {e}, using defaults")
        return default_prompts

# Load system prompts at module level
SYSTEM_PROMPTS = load_system_prompts()

class SessionManager:
    """Manages persistent chat sessions"""
    
    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.session_timeout = 24 * 60 * 60  # 24 hours in seconds
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())[:8]  # Short session ID
        self._save_session(session_id, {
            'history': [],
            'created_at': time.time(),
            'last_accessed': time.time(),
            'settings': {
                'system_prompt': 'Default Assistant',
                'custom_prompt': '',
                'temperature': 0.3,
                'max_tokens': 4192
            }
        })
        print(f"📂 Created new session: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> dict:
        """Load session data or create new if doesn't exist"""
        if not session_id:
            return self._create_empty_session()
        
        session_file = self.sessions_dir / f"session_{session_id}.pkl"
        
        if not session_file.exists():
            print(f"⚠️ Session {session_id} not found, creating new")
            return self._create_empty_session()
        
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Check if session expired
            if time.time() - session_data.get('last_accessed', 0) > self.session_timeout:
                print(f"⏰ Session {session_id} expired, creating new")
                session_file.unlink(missing_ok=True)  # Delete expired session
                return self._create_empty_session()
            
            # Update last accessed time
            session_data['last_accessed'] = time.time()
            self._save_session(session_id, session_data)
            
            print(f"📂 Loaded session: {session_id} ({len(session_data['history'])} messages)")
            return session_data
            
        except Exception as e:
            print(f"❌ Error loading session {session_id}: {e}")
            return self._create_empty_session()
    
    def save_session(self, session_id: str, history: list, settings: dict = None) -> None:
        """Save session data"""
        if not session_id:
            return
        
        session_data = {
            'history': history,
            'last_accessed': time.time(),
            'settings': settings or {}
        }
        
        # Load existing session to preserve created_at
        existing = self.load_session(session_id)
        if 'created_at' in existing:
            session_data['created_at'] = existing['created_at']
        else:
            session_data['created_at'] = time.time()
        
        self._save_session(session_id, session_data)
    
    def _save_session(self, session_id: str, session_data: dict) -> None:
        """Internal method to save session data"""
        session_file = self.sessions_dir / f"session_{session_id}.pkl"
        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
        except Exception as e:
            print(f"❌ Error saving session {session_id}: {e}")
    
    def _create_empty_session(self) -> dict:
        """Create empty session data structure"""
        return {
            'history': [],
            'created_at': time.time(),
            'last_accessed': time.time(),
            'settings': {
                'system_prompt': 'Default Assistant',
                'custom_prompt': '',
                'temperature': 0.3,
                'max_tokens': 4192
            }
        }
    
    def list_sessions(self) -> list:
        """List all active sessions"""
        sessions = []
        for session_file in self.sessions_dir.glob("session_*.pkl"):
            try:
                session_id = session_file.stem.replace("session_", "")
                with open(session_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Skip expired sessions
                if time.time() - data.get('last_accessed', 0) > self.session_timeout:
                    session_file.unlink(missing_ok=True)
                    continue
                
                sessions.append({
                    'id': session_id,
                    'messages': len(data['history']),
                    'created': data.get('created_at', 0),
                    'accessed': data.get('last_accessed', 0)
                })
            except:
                continue
        
        return sorted(sessions, key=lambda x: x['accessed'], reverse=True)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count"""
        cleaned = 0
        for session_file in self.sessions_dir.glob("session_*.pkl"):
            try:
                with open(session_file, 'rb') as f:
                    data = pickle.load(f)
                
                if time.time() - data.get('last_accessed', 0) > self.session_timeout:
                    session_file.unlink(missing_ok=True)
                    cleaned += 1
            except:
                session_file.unlink(missing_ok=True)
                cleaned += 1
        
        return cleaned

class MetricsCollector:
    """Time series metrics collector with visualization support and persistent storage"""
    
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.metrics_data = defaultdict(lambda: deque(maxlen=max_points))
        self.timestamps = deque(maxlen=max_points)
        self.pull_interval = 30  # Default 30 seconds
        self.collection_active = False
        self.collection_thread = None
        self.lock = threading.Lock()
        
        # Archive settings
        self.archive_dir = Path("metrics_archive")
        self.archive_dir.mkdir(exist_ok=True)
        self.archive_file = self.archive_dir / "metrics_data.json"
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 60 seconds
        
        # Load existing data on startup
        self.load_archive()
        
        # Metric categories and their color schemes
        self.metric_categories = {
            'memory': {
                'color': '#FF6B6B',  # Red
                'patterns': [
                    r'memory_usage_bytes',
                    r'memory_allocated_bytes', 
                    r'memory_cached_bytes',
                    r'memory_free_bytes',
                    r'gpu_memory_usage',
                    r'process_virtual_memory_bytes',
                    r'process_resident_memory_bytes'
                ]
            },
            'transactions': {
                'color': '#4ECDC4',  # Teal
                'patterns': [
                    r'http_requests_total',
                    r'http_request_duration.*seconds',
                    r'requests_per_second',
                    r'response_time_seconds',
                    r'active_requests',
                    r'queue_size',
                    r'http_request_size_bytes'
                ]
            },
            'tokens': {
                'color': '#45B7D1',  # Blue
                'patterns': [
                    r'prompt_tokens_total',
                    r'completion_tokens_total',
                    r'tokens_per_second',
                    r'input_token_count',
                    r'output_token_count',
                    r'tokenizer_calls_total',
                    r'vllm:prompt_tokens.*',
                    r'vllm:generation_tokens.*',
                    r'vllm:.*tokens.*',
                    r'.*tokens.*'
                ]
            },
            'model': {
                'color': '#96CEB4',  # Green
                'patterns': [
                    r'model_load_time_seconds',
                    r'inference_time_seconds',
                    r'model_memory_usage',
                    r'batch_size',
                    r'concurrent_requests',
                    r'vllm:request.*time.*seconds',
                    r'vllm:.*inference.*',
                    r'vllm:.*latency.*',
                    r'vllm:e2e.*',
                    r'vllm:time_.*',
                    r'.*inference.*',
                    r'.*latency.*',
                    r'.*prefill.*',
                    r'.*decode.*'
                ]
            }
        }
    
    def parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format and extract current values"""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse metric line: metric_name{labels} value [timestamp]
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name_with_labels = parts[0]
                    try:
                        value = float(parts[1])
                        
                        # Extract metric name (before any '{' or ' ')
                        metric_name = metric_name_with_labels.split('{')[0]
                        metrics[metric_name] = value
                    except ValueError:
                        continue
                        
        return metrics
    
    def categorize_metric(self, metric_name: str) -> str:
        """Categorize a metric based on its name"""
        for category, info in self.metric_categories.items():
            for pattern in info['patterns']:
                if re.search(pattern, metric_name, re.IGNORECASE):
                    return category
        return 'other'
    
    def debug_categorization(self):
        """Debug categorization of all current metrics"""
        with self.lock:
            categorized = {'memory': [], 'transactions': [], 'tokens': [], 'model': [], 'other': []}
            
            print(f"\n🔍 CATEGORIZATION DEBUG:")
            print(f"Total metrics to categorize: {len(self.metrics_data)}")
            
            for metric_name in self.metrics_data.keys():
                category = self.categorize_metric(metric_name)
                categorized[category].append(metric_name)
            
            for category, metrics in categorized.items():
                print(f"  {category.upper()}: {len(metrics)} metrics")
                for metric in metrics[:3]:  # Show first 3 examples
                    print(f"    - {metric}")
                if len(metrics) > 3:
                    print(f"    ... and {len(metrics) - 3} more")
            
            return categorized
    
    def add_metrics_data(self, metrics_text: str):
        """Add new metrics data point"""
        if not metrics_text:
            return
            
        with self.lock:
            timestamp = datetime.now()
            parsed_metrics = self.parse_prometheus_metrics(metrics_text)
            
            self.timestamps.append(timestamp)
            
            for metric_name, value in parsed_metrics.items():
                self.metrics_data[metric_name].append(value)
    
    def get_metrics_by_category(self, category: str) -> Dict[str, List]:
        """Get time series data for a specific category"""
        with self.lock:
            category_metrics = {}
            timestamps = list(self.timestamps)
            
            print(f"🔍 Looking for {category} metrics from {len(self.metrics_data)} total metrics")
            print(f"📊 Available timestamps: {len(timestamps)}")
            
            for metric_name, values in self.metrics_data.items():
                metric_category = self.categorize_metric(metric_name)
                if metric_category == category:
                    print(f"  ✅ Found {category} metric: {metric_name} ({len(values)} values)")
                    category_metrics[metric_name] = {
                        'values': list(values),
                        'timestamps': timestamps[-len(values):] if values else timestamps[:len(values)] if timestamps else []
                    }
            
            print(f"🎯 Found {len(category_metrics)} {category} metrics")
            return category_metrics
    
    def create_time_series_plot(self, category: str) -> Optional[go.Figure]:
        """Create a plotly time series plot for a category"""
        try:
            print(f"\n🔧 Starting plot creation for {category}")
            
            if not PLOTTING_AVAILABLE:
                print(f"⚠️ Plotting not available for {category}")
                return None
                
            metrics_data = self.get_metrics_by_category(category)
            print(f"📊 Creating plot for {category}, found {len(metrics_data)} metrics")
            
            if not metrics_data:
                print(f"⚠️ No metrics data for {category} - returning None")
                return None
                
            fig = go.Figure()
            print(f"✅ Created empty figure for {category}")
            
            category_color = self.metric_categories.get(category, {}).get('color', '#999999')
            traces_added = 0
            
            for i, (metric_name, data) in enumerate(metrics_data.items()):
                values = data['values']
                timestamps = data['timestamps']
                
                print(f"  Processing {metric_name}: {len(values)} values, {len(timestamps)} timestamps")
                
                if values:
                    # If we have values but no timestamps, create synthetic ones
                    if not timestamps or len(timestamps) != len(values):
                        base_time = datetime.now()
                        timestamps = [base_time - timedelta(seconds=(len(values)-1-i)*30) for i in range(len(values))]
                        print(f"  Generated {len(timestamps)} synthetic timestamps")
                    
                    # Use variations of the category color
                    color_variant = self._get_color_variant(category_color, i)
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=metric_name,
                        line=dict(color=color_variant, width=2),
                        marker=dict(size=4)
                    ))
                    traces_added += 1
                    print(f"  ✅ Added trace for {metric_name}")
            
            if traces_added == 0:
                print(f"⚠️ No traces added to {category} plot")
                return None
            
            fig.update_layout(
                title=f'{category.title()} Metrics Over Time ({traces_added} metrics)',
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            print(f"✅ Created {category} plot with {traces_added} traces")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating plot for {category}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _get_color_variant(self, base_color: str, index: int) -> str:
        """Generate color variants for multiple metrics in same category"""
        # Simple color variation by adjusting opacity and hue
        variants = [base_color, base_color + '80', base_color + '60', base_color + 'CC']
        return variants[index % len(variants)]
    
    def set_pull_interval(self, seconds: int):
        """Set the metrics collection interval"""
        self.pull_interval = max(5, seconds)  # Minimum 5 seconds
    
    def start_collection(self, client):
        """Start automated metrics collection"""
        if self.collection_active:
            return
            
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, 
            args=(client,), 
            daemon=True
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop automated metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self, client):
        """Background thread for collecting metrics"""
        while self.collection_active:
            try:
                metrics_result = client.get_metrics()
                if metrics_result.get('success'):
                    self.add_metrics_data(metrics_result.get('data', ''))
                    
                    # Periodically save to archive
                    current_time = time.time()
                    if current_time - self.last_save_time > self.save_interval:
                        self.save_archive()
                        self.last_save_time = current_time
                        
            except Exception as e:
                print(f"⚠️ Metrics collection error: {e}")
                
            time.sleep(self.pull_interval)
        
        # Save final state when collection stops
        print("💾 Collection stopped, saving final archive...")
        self.save_archive()
    
    def save_archive(self):
        """Save current metrics data to persistent storage"""
        try:
            with self.lock:
                # Convert deque objects to lists for JSON serialization
                archive_data = {
                    'metrics_data': {
                        name: list(values) for name, values in self.metrics_data.items()
                    },
                    'timestamps': [ts.isoformat() for ts in self.timestamps],
                    'last_updated': datetime.now().isoformat(),
                    'pull_interval': self.pull_interval,
                    'max_points': self.max_points
                }
                
                # Create backup of existing archive
                if self.archive_file.exists():
                    backup_file = self.archive_dir / f"metrics_backup_{int(time.time())}.json"
                    self.archive_file.rename(backup_file)
                    
                    # Keep only last 5 backups
                    backups = sorted(self.archive_dir.glob("metrics_backup_*.json"))
                    for old_backup in backups[:-5]:
                        old_backup.unlink(missing_ok=True)
                
                # Save current data
                with open(self.archive_file, 'w') as f:
                    json.dump(archive_data, f, indent=2)
                
                print(f"📦 Metrics archive saved: {len(self.metrics_data)} metrics, {len(self.timestamps)} timestamps")
                
        except Exception as e:
            print(f"❌ Error saving metrics archive: {str(e)}")
    
    def load_archive(self):
        """Load metrics data from persistent storage"""
        try:
            if not self.archive_file.exists():
                print("📦 No metrics archive found, starting fresh")
                return
            
            with open(self.archive_file, 'r') as f:
                archive_data = json.load(f)
            
            # Load metrics data
            for metric_name, values in archive_data.get('metrics_data', {}).items():
                self.metrics_data[metric_name] = deque(values, maxlen=self.max_points)
            
            # Load timestamps
            timestamp_strings = archive_data.get('timestamps', [])
            self.timestamps = deque([
                datetime.fromisoformat(ts) for ts in timestamp_strings
            ], maxlen=self.max_points)
            
            # Update settings from archive
            self.pull_interval = archive_data.get('pull_interval', self.pull_interval)
            
            last_updated = archive_data.get('last_updated')
            loaded_count = len(self.metrics_data)
            timestamps_count = len(self.timestamps)
            
            print(f"✅ Metrics archive loaded: {loaded_count} metrics, {timestamps_count} timestamps")
            if last_updated:
                print(f"📅 Last updated: {last_updated}")
                
        except Exception as e:
            print(f"❌ Error loading metrics archive: {str(e)}")
            print("📦 Starting with empty metrics collection")
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics data to a file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_export_{timestamp}.json"
            
            export_path = self.archive_dir / filename
            
            with self.lock:
                export_data = {
                    'exported_at': datetime.now().isoformat(),
                    'export_version': '1.0',
                    'metrics_count': len(self.metrics_data),
                    'timestamps_count': len(self.timestamps),
                    'pull_interval': self.pull_interval,
                    'metrics_data': {
                        name: list(values) for name, values in self.metrics_data.items()
                    },
                    'timestamps': [ts.isoformat() for ts in self.timestamps],
                    'categories': list(self.metric_categories.keys())
                }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return f"✅ Metrics exported to {export_path}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def import_metrics(self, filename: str) -> str:
        """Import metrics data from a file"""
        try:
            import_path = self.archive_dir / filename
            
            if not import_path.exists():
                return f"❌ File not found: {filename}"
            
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Merge with existing data
            with self.lock:
                for metric_name, values in import_data.get('metrics_data', {}).items():
                    # Merge values, keeping newest points
                    existing_values = list(self.metrics_data[metric_name])
                    combined_values = existing_values + values
                    self.metrics_data[metric_name] = deque(combined_values[-self.max_points:], maxlen=self.max_points)
                
                # Merge timestamps
                import_timestamps = [
                    datetime.fromisoformat(ts) for ts in import_data.get('timestamps', [])
                ]
                existing_timestamps = list(self.timestamps)
                combined_timestamps = existing_timestamps + import_timestamps
                # Sort by time and take most recent points
                combined_timestamps.sort()
                self.timestamps = deque(combined_timestamps[-self.max_points:], maxlen=self.max_points)
            
            # Save merged data
            self.save_archive()
            
            imported_count = len(import_data.get('metrics_data', {}))
            return f"✅ Imported {imported_count} metrics from {filename}"
            
        except Exception as e:
            return f"❌ Import failed: {str(e)}"

class StreamingResponse:
    """Handle streaming response accumulation"""
    def __init__(self):
        self.content = ""
        self.is_complete = False
        self.error = None

class ChatClient:
    """Enhanced chat client with streaming and timeout handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create optimized session with proper settings"""
        session = requests.Session()
        session.verify = self.config.verify_ssl
        
        # Optimize connection settings
        session.headers.update({
            'User-Agent': 'OpenShift-AI-Chat/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=120, max=100'
        })
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # We handle retries manually
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def test_connection(self) -> Dict[str, Any]:
        """Comprehensive connection diagnostics"""
        results = {}
        
        # Test 1: Basic health check
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/health", 
                timeout=self.config.connect_timeout
            )
            results['health'] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'response': response.text[:200],
                'latency': response.elapsed.total_seconds()
            }
        except Exception as e:
            results['health'] = {'success': False, 'error': str(e)}
        
        # Test 2: Available models
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/v1/models", 
                timeout=self.config.connect_timeout
            )
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                results['models'] = {
                    'success': True,
                    'available': available_models,
                    'configured': self.config.model_name,
                    'match': self.config.model_name in available_models
                }
            else:
                results['models'] = {
                    'success': False, 
                    'status': response.status_code,
                    'response': response.text[:200]
                }
        except Exception as e:
            results['models'] = {'success': False, 'error': str(e)}
        
        # Test 3: Simple chat completion (non-streaming)
        try:
            test_payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.1,
                "stream": False
            }
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_endpoint}/v1/chat/completions",
                json=test_payload,
                timeout=self.config.connect_timeout
            )
            latency = time.time() - start_time
            
            results['chat_test'] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'latency': latency,
                'response': response.text[:200] if response.status_code != 200 else "OK"
            }
        except Exception as e:
            results['chat_test'] = {'success': False, 'error': str(e)}
        
        # Test 4: Streaming test
        try:
            test_payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 20,
                "temperature": 0.1,
                "stream": True
            }
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_endpoint}/v1/chat/completions",
                json=test_payload,
                stream=True,
                timeout=(self.config.connect_timeout, 60)
            )
            
            if response.status_code == 200:
                # Test streaming response
                content = ""
                for line in response.iter_lines():
                    if line and len(content) < 50:  # Limit test
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line[6:] != "[DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if 'choices' in chunk:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content += delta['content']
                            except:
                                continue
                        if content:  # Got some content, test passes
                            break
                
                latency = time.time() - start_time
                results['streaming_test'] = {
                    'success': True,
                    'latency': latency,
                    'content_received': len(content) > 0
                }
            else:
                results['streaming_test'] = {
                    'success': False,
                    'status': response.status_code,
                    'response': response.text[:200]
                }
        except Exception as e:
            results['streaming_test'] = {'success': False, 'error': str(e)}
        
        return results
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get API version information"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/version", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/metrics", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.text  # Metrics are usually in Prometheus format
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Get OpenAPI specification"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/openapi.json", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _estimate_context_size(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total context size in characters"""
        return sum(len(msg.get('content', '')) for msg in messages)
    
    def _should_use_streaming(self, messages: List[Dict[str, str]], max_tokens: int) -> bool:
        """Determine if streaming should be used based on context size"""
        context_size = self._estimate_context_size(messages)
        
        # Use streaming for:
        # 1. Large contexts (>4000 chars)
        # 2. Large max_tokens (>1000)
        # 3. Multiple history messages (>3)
        return (
            context_size > self.config.auto_stream_threshold or
            max_tokens > 1000 or
            len(messages) > 4
        )
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.6,
        max_tokens: int = 2048,
        force_streaming: bool = False
    ) -> str:
        """Smart chat completion with fallback to non-streaming"""
        
        # Decide on streaming
        use_streaming = force_streaming or self._should_use_streaming(messages, max_tokens)
        context_size = self._estimate_context_size(messages)
        
        print(f"🧠 Context size: {context_size} chars, Streaming: {use_streaming}")
        
        if use_streaming:
            print("🌊 Trying streaming first...")
            streaming_result = self._stream_completion(messages, temperature, max_tokens)
            
            # If streaming failed but didn't return an error message, try direct
            if streaming_result.startswith("❌") and context_size < 8000:
                print("🔄 Streaming failed, trying direct completion as fallback...")
                return self._direct_completion(messages, temperature, min(max_tokens, 1000))
            else:
                return streaming_result
        else:
            return self._direct_completion(messages, temperature, max_tokens)
    
    def _direct_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> str:
        """Direct completion for small contexts"""
        url = f"{self.config.api_endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        print(f"🔄 Direct request to: {url}")
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                timeout = (self.config.connect_timeout, self.config.read_timeout)
                response = self.session.post(url, json=payload, timeout=timeout)
                
                print(f"📡 Response: {response.status_code}")
                
                if response.status_code == 504:
                    print("⏰ Gateway timeout - switching to streaming")
                    return self._stream_completion(messages, temperature, max_tokens)
                elif response.status_code != 200:
                    error_detail = response.text[:300]
                    if attempt < self.config.max_retry_attempts - 1:
                        print(f"❌ Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return f"API Error {response.status_code}: {error_detail}"
                
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print(f"✅ Success! Length: {len(content)}")
                    return content
                else:
                    return f"Unexpected response format: {result}"
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retry_attempts - 1:
                    print("🔄 Retrying with streaming...")
                    return self._stream_completion(messages, temperature, max_tokens)
                else:
                    return "❌ Request timed out. Try reducing context size or message length."
            except Exception as e:
                if attempt < self.config.max_retry_attempts - 1:
                    print(f"💥 Attempt {attempt + 1} error: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return f"❌ Error: {str(e)}"
        
        return "❌ All retry attempts failed"
    
    def _stream_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> str:
        """Debug-enhanced streaming completion with detailed logging"""
        url = f"{self.config.api_endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        print(f"🌊 DEBUG: Streaming request to: {url}")
        print(f"📦 DEBUG: Payload keys: {list(payload.keys())}")
        
        for attempt in range(self.config.max_retry_attempts):
            accumulated = ""
            try:
                print(f"🚀 DEBUG: Attempt {attempt + 1} starting...")
                
                response = self.session.post(
                    url, 
                    json=payload, 
                    stream=True, 
                    timeout=(self.config.connect_timeout, None)
                )
                
                print(f"📡 DEBUG: Response status: {response.status_code}")
                print(f"📋 DEBUG: Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    error_detail = response.text[:500]
                    print(f"❌ DEBUG: Error response: {error_detail}")
                    if attempt < self.config.max_retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return f"❌ Streaming Error {response.status_code}: {error_detail}"
                
                # Process streaming response with extensive debugging
                start_time = time.time()
                chunk_count = 0
                line_count = 0
                data_lines = 0
                content_chunks = 0
                
                print("🌊 DEBUG: Starting to process streaming response...")
                
                try:
                    for line in response.iter_lines():
                        line_count += 1
                        current_time = time.time()
                        
                        # Timeout protection
                        if current_time - start_time > self.config.streaming_timeout:
                            print(f"⏰ DEBUG: Timeout after {self.config.streaming_timeout}s")
                            if accumulated:
                                return accumulated + "\n\n[Truncated: timeout]"
                            break
                        
                        if line:
                            # Decode line
                            try:
                                line_str = line.decode('utf-8')
                            except UnicodeDecodeError:
                                print(f"⚠️ DEBUG: Unicode decode error for line {line_count}")
                                continue
                            
                            # Log every 100th line for debugging
                            if line_count % 100 == 0:
                                print(f"🔍 DEBUG: Processed {line_count} lines, {data_lines} data lines, {content_chunks} content chunks")
                            
                            # Process data lines
                            if line_str.startswith("data: "):
                                data_lines += 1
                                data = line_str[6:].strip()
                                
                                # Debug first few data lines
                                if data_lines <= 5:
                                    print(f"📄 DEBUG: Data line {data_lines}: {data[:100]}...")
                                
                                if data == "[DONE]":
                                    print("✅ DEBUG: Received [DONE] signal")
                                    break
                                    
                                if data and data != "":
                                    try:
                                        chunk = json.loads(data)
                                        chunk_count += 1
                                        
                                        # Debug chunk structure
                                        if chunk_count <= 3:
                                            print(f"🧩 DEBUG: Chunk {chunk_count} keys: {list(chunk.keys())}")
                                            if 'choices' in chunk:
                                                print(f"🧩 DEBUG: Choices[0] keys: {list(chunk['choices'][0].keys())}")
                                        
                                        if 'choices' in chunk and len(chunk['choices']) > 0:
                                            choice = chunk['choices'][0]
                                            delta = choice.get('delta', {})
                                            
                                            if 'content' in delta:
                                                content_piece = delta.get('content', '')
                                                if content_piece:  # Only add non-empty content
                                                    accumulated += content_piece
                                                    content_chunks += 1
                                                    
                                                    # Progress every 500 chars
                                                    if len(accumulated) % 500 == 0:
                                                        print(f"📝 DEBUG: {len(accumulated)} chars, {content_chunks} content chunks")
                                            
                                            # Check for finish
                                            if choice.get('finish_reason'):
                                                finish_reason = choice.get('finish_reason')
                                                print(f"🏁 DEBUG: Finished with reason: {finish_reason}")
                                                break
                                                
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️ DEBUG: JSON error on line {data_lines}: {str(e)}")
                                        print(f"⚠️ DEBUG: Problematic data: {data[:200]}")
                                        continue
                                    except Exception as e:
                                        print(f"⚠️ DEBUG: Chunk processing error: {str(e)}")
                                        continue
                    
                    # Final debug summary
                    elapsed = time.time() - start_time
                    print(f"🏁 DEBUG: Stream complete after {elapsed:.1f}s")
                    print(f"📊 DEBUG: {line_count} lines, {data_lines} data lines, {chunk_count} chunks, {content_chunks} content chunks")
                    print(f"📏 DEBUG: Final accumulated length: {len(accumulated)}")
                    
                    if accumulated:
                        print(f"✅ DEBUG: SUCCESS! Returning {len(accumulated)} chars")
                        print(f"📖 DEBUG: First 200 chars: {accumulated[:200]}")
                        return accumulated
                    else:
                        print("📭 DEBUG: No content accumulated!")
                        print(f"📊 DEBUG: Had {chunk_count} chunks but {content_chunks} content chunks")
                        
                        if attempt < self.config.max_retry_attempts - 1:
                            print(f"🔄 DEBUG: Retrying attempt {attempt + 2}...")
                            time.sleep(2)
                            continue
                        return f"❌ No content after processing {chunk_count} chunks"
                        
                except Exception as stream_error:
                    print(f"💥 DEBUG: Stream processing exception: {stream_error}")
                    print(f"🔍 DEBUG: Exception traceback: {traceback.format_exc()}")
                    
                    if accumulated:
                        print(f"💾 DEBUG: Returning {len(accumulated)} partial chars")
                        return accumulated + f"\n\n[Stream error: {stream_error}]"
                    
                    if attempt < self.config.max_retry_attempts - 1:
                        continue
                    return f"❌ Stream processing failed: {stream_error}"
                
                finally:
                    try:
                        response.close()
                    except:
                        pass
                    
            except requests.exceptions.Timeout:
                print(f"⏰ DEBUG: Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "❌ Connection timed out"
                
            except Exception as e:
                print(f"💥 DEBUG: Request exception on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"❌ Request failed: {str(e)}"
        
        return f"❌ All {self.config.max_retry_attempts} attempts failed"
    
    def health_check(self) -> bool:
        """Simple health check"""
        try:
            results = self.test_connection()
            return results.get('health', {}).get('success', False)
        except:
            return False

class ChatInterface:
    """Enhanced chat interface with smart processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = ChatClient(config)
        self.system_prompts = SYSTEM_PROMPTS.copy()  # Keep a local copy
        self._processing = False  # Flag to prevent double processing
        self.session_manager = SessionManager()  # Add session management
        self.metrics_collector = MetricsCollector()  # Add metrics collection
    
    def process_message(
        self,
        message: str,
        history: List[List[str]],
        system_prompt: str,
        custom_prompt: str,
        temperature: float,
        max_tokens: int,
        uploaded_file: Optional[Any] = None,
        session_id: str = None
    ) -> Tuple[str, List[List[str]], Optional[Any], str]:
        """Enhanced message processing with UI debugging"""
        
        if not message.strip():
            return "", history, None, session_id or ""
        
        # Create or get session ID
        if not session_id:
            session_id = self.session_manager.create_session()
        
        # Prevent double processing
        if self._processing:
            print("⚠️ Already processing, ignoring duplicate request")
            return "", history, None, session_id
        
        self._processing = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🚀 PROCESSING: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"🌡️ TEMPERATURE: {temperature}")
            print(f"📏 MAX_TOKENS: {max_tokens}")
            print(f"🎯 SYSTEM_PROMPT: {system_prompt}")
            print(f"{'='*60}")
            
            # Build system prompt
            active_system_prompt = custom_prompt if custom_prompt.strip() else self.system_prompts.get(system_prompt, "")
            
            # Build messages list
            messages = []
            if active_system_prompt:
                messages.append({"role": "system", "content": active_system_prompt})
            
            # Add conversation history (limit to prevent huge contexts)
            recent_history = history[-10:] if len(history) > 10 else history
            for user_msg, assistant_msg in recent_history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # Handle file upload with size limits (only for current message)
            if uploaded_file is not None:
                file_content = self._process_file(uploaded_file)
                message = f"{message}\n\n[File content]:\n{file_content}"
                print(f"📎 File attached: {getattr(uploaded_file, 'name', 'unknown')} ({len(file_content)} chars)")
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Context analysis
            context_size = sum(len(msg.get('content', '')) for msg in messages)
            print(f"📊 Context: {context_size} chars, Temp: {temperature}, Tokens: {max_tokens}")
            
            if context_size > 20000:
                error_response = "❌ Context too large. Please start a new conversation or upload a smaller file."
                print(f"❌ Context too large: {context_size} chars")
                new_history = history + [[message, error_response]]
                # Save session with error
                self.session_manager.save_session(session_id, new_history, {
                    'system_prompt': system_prompt,
                    'custom_prompt': custom_prompt,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                })
                print(f"🔄 UI DEBUG: Returning history with {len(new_history)} items")
                return "", new_history, None, session_id
            
            # Process with smart completion
            print("🤖 Calling chat completion...")
            response = self.client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"✅ Response received: {len(response)} chars")
            print(f"📝 Response preview: {response[:200]}...")
            
            # Ensure response is a string and not empty
            if not isinstance(response, str):
                response = str(response)
            
            # Clean up model reasoning tags
            if response.startswith('<think>') and '</think>' in response:
                # Extract content after thinking tags
                think_end = response.find('</think>')
                if think_end != -1:
                    response = response[think_end + 8:].strip()
                    print(f"🧠 Cleaned thinking tags, new length: {len(response)}")
            
            # Remove any remaining HTML-like tags that might interfere
            import re
            response = re.sub(r'<[^>]+>', '', response).strip()
            
            if not response.strip():
                response = "❌ Empty response received from model after cleaning"
                print("⚠️ Empty response after cleaning")
            
            print(f"📝 Final cleaned response preview: {response[:200]}...")
            
            # Create new history entry
            new_history = history + [[message, response]]
            print(f"🔄 UI DEBUG: Creating history entry:")
            print(f"   - User message: {len(message)} chars")
            print(f"   - Assistant response: {len(response)} chars")
            print(f"   - Total history items: {len(new_history)}")
            
            # Validate the history structure
            try:
                last_entry = new_history[-1]
                if len(last_entry) != 2:
                    print(f"⚠️ Invalid history entry format: {last_entry}")
                else:
                    print(f"✅ Valid history entry: [{len(last_entry[0])} chars, {len(last_entry[1])} chars]")
            except Exception as e:
                print(f"❌ History validation error: {e}")
            
            # Save successful session
            self.session_manager.save_session(session_id, new_history, {
                'system_prompt': system_prompt,
                'custom_prompt': custom_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
            
            print(f"🔄 UI DEBUG: Returning empty message and {len(new_history)} history items")
            print(f"💾 Session {session_id} saved with {len(new_history)} messages")
            return "", new_history, None, session_id
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            print(f"💥 Exception: {error_msg}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            
            new_history = history + [[message, error_msg]]
            # Save session with error
            self.session_manager.save_session(session_id, new_history, {
                'system_prompt': system_prompt,
                'custom_prompt': custom_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
            print(f"🔄 UI DEBUG: Error case - returning {len(new_history)} history items")
            return "", new_history, None, session_id
        
        finally:
            self._processing = False
    
    def load_session(self, session_id: str) -> Tuple[List[List[str]], str, str, float, int]:
        """Load session and return history and settings"""
        if not session_id:
            return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens
        
        session_data = self.session_manager.load_session(session_id)
        settings = session_data.get('settings', {})
        
        return (
            session_data.get('history', []),
            settings.get('system_prompt', 'Default Assistant'),
            settings.get('custom_prompt', ''),
            settings.get('temperature', self.config.default_temperature),
            settings.get('max_tokens', self.config.default_max_tokens)
        )
    
    def new_session(self) -> Tuple[List[List[str]], str]:
        """Create a new session"""
        session_id = self.session_manager.create_session()
        return [], session_id
    
    def get_session_list(self) -> Tuple[str, List[str]]:
        """Get formatted list of sessions and their IDs"""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            return "No active sessions found.", []
        
        session_text = "# 📂 Active Sessions\n\n"
        session_ids = []
        
        for i, session in enumerate(sessions[:10]):  # Show last 10 sessions
            created = datetime.fromtimestamp(session['created']).strftime('%Y-%m-%d %H:%M')
            accessed = datetime.fromtimestamp(session['accessed']).strftime('%Y-%m-%d %H:%M')
            session_text += f"**[{i+1}] Session `{session['id']}`**\n"
            session_text += f"- Messages: {session['messages']}\n"
            session_text += f"- Created: {created}\n"
            session_text += f"- Last accessed: {accessed}\n\n"
            session_ids.append(session['id'])
        
        return session_text, session_ids
    
    def _process_file(self, file) -> str:
        """Process uploaded file with size limits"""
        try:
            if hasattr(file, 'name'):
                file_path = file.name
                file_ext = file_path.lower().split('.')[-1]
                
                if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
                    return f"[Image: {file_path.split('/')[-1]}] Note: Describe what you want to know about this image."
                elif file_ext in ['pdf', 'xlsx', 'docx']:
                    return f"[Binary file: {file_path.split('/')[-1]}] Note: Please convert to text or describe the content."
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Strict size limit to prevent timeouts
                        if len(content) > self.config.max_file_chars:
                            content = content[:self.config.max_file_chars] + f"\n\n[File truncated to {self.config.max_file_chars} chars to prevent timeouts]"
                        return content
            return "Unable to read file"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def clear_history(self) -> List:
        """Clear conversation history"""
        return []
    
    def export_conversation(self, history: List[List[str]]) -> str:
        """Export conversation history"""
        if not history:
            return "No conversation to export."
        
        export_text = f"# Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, (user, assistant) in enumerate(history):
            export_text += f"## Message {i+1}\n\n"
            export_text += f"**User:** {user}\n\n"
            export_text += f"**Assistant:** {assistant}\n\n"
            export_text += "---\n\n"
        
        return export_text
    
    def test_simple_streaming(self) -> str:
        """Test streaming with a simple request"""
        print("🧪 Testing simple streaming...")
        try:
            test_messages = [{"role": "user", "content": "Count from 1 to 5"}]
            response = self.client._stream_completion(test_messages, 0.1, 50)
            return f"# 🧪 Streaming Test Results\n\n**Response:** {response}\n\n**Length:** {len(response)} characters"
        except Exception as e:
            return f"# 🧪 Streaming Test Results\n\n**Status:** FAILED\n**Error:** {str(e)}"
    
    def test_ui_update(self, history: List[List[str]]) -> Tuple[str, List[List[str]], Optional[Any]]:
        """Test UI update with a sample response"""
        print("🧪 Testing UI update...")
        test_response = "This is a test response to verify UI updates are working correctly. ✅"
        test_message = "UI Test"
        
        new_history = history + [[test_message, test_response]]
        print(f"🔄 UI TEST: Added entry, total items: {len(new_history)}")
        
        return "", new_history, None
    
    def reload_system_prompts(self) -> str:
        """Reload system prompts from file"""
        try:
            self.system_prompts = load_system_prompts()
            return f"✅ Successfully reloaded {len(self.system_prompts)} system prompts"
        except Exception as e:
            return f"❌ Error reloading prompts: {str(e)}"
    
    def save_custom_prompt(self, prompt_name: str, prompt_content: str) -> str:
        """Save a custom prompt to the file"""
        if not prompt_name or not prompt_content:
            return "❌ Please provide both prompt name and content"
        
        try:
            prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
            
            # Load current prompts
            current_prompts = {}
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    current_prompts = json.load(f)
            
            # Add/update the prompt
            action = "updated" if prompt_name in current_prompts else "created"
            current_prompts[prompt_name] = prompt_content
            
            # Save back to file
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(current_prompts, f, indent=4, ensure_ascii=False)
            
            # Reload prompts
            self.system_prompts = load_system_prompts()
            
            return f"✅ Successfully {action} prompt '{prompt_name}'"
        except Exception as e:
            return f"❌ Error saving prompt: {str(e)}"
    
    def delete_prompt(self, prompt_name: str) -> str:
        """Delete a prompt from the file"""
        if not prompt_name:
            return "❌ Please select a prompt to delete"
        
        if prompt_name in ["Default Assistant", "Technical Expert", "Code Assistant"]:
            return "❌ Cannot delete core system prompts"
        
        try:
            prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
            
            # Load current prompts
            current_prompts = {}
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    current_prompts = json.load(f)
            
            if prompt_name not in current_prompts:
                return f"❌ Prompt '{prompt_name}' not found"
            
            # Remove the prompt
            del current_prompts[prompt_name]
            
            # Save back to file
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(current_prompts, f, indent=4, ensure_ascii=False)
            
            # Reload prompts
            self.system_prompts = load_system_prompts()
            
            return f"✅ Successfully deleted prompt '{prompt_name}'"
        except Exception as e:
            return f"❌ Error deleting prompt: {str(e)}"
    
    def load_prompt_for_editing(self, prompt_name: str) -> tuple:
        """Load a prompt for editing"""
        if not prompt_name or prompt_name not in self.system_prompts:
            return "", ""
        
        return prompt_name, self.system_prompts[prompt_name]
    
    def get_management_overview(self) -> str:
        """Get comprehensive management overview"""
        overview = "# 🎛️ Model Serving Management\n"
        overview += f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        try:
            # Get all available information with error handling
            health_result = self.client.test_connection()
        except Exception as e:
            health_result = {'error': str(e)}
            
        try:
            version_info = self.client.get_version_info()
        except Exception as e:
            version_info = {'success': False, 'error': str(e)}
            
        try:
            metrics_info = self.client.get_metrics()
        except Exception as e:
            metrics_info = {'success': False, 'error': str(e)}
        
        # Model Information Section
        overview += "## 📊 Model Information\n"
        models_data = health_result.get('models', {})
        if models_data.get('success'):
            overview += f"**Status:** 🟢 Online\n"
            overview += f"**Available Models:** {', '.join(models_data.get('available', []))}\n"
            overview += f"**Active Model:** {models_data.get('configured', 'Unknown')}\n"
        else:
            overview += "**Status:** 🔴 Offline or Unavailable\n"
        
        # Server Health Section
        overview += "\n## 🏥 Server Health\n"
        health_data = health_result.get('health', {})
        if health_data.get('success'):
            overview += f"**Health Status:** ✅ Healthy\n"
            overview += f"**Response Time:** {health_data.get('latency', 0)*1000:.0f}ms\n"
        else:
            overview += f"**Health Status:** ❌ Unhealthy\n"
            overview += f"**Error:** {health_data.get('error', 'Unknown')}\n"
        
        # API Version Section
        overview += "\n## 🔧 API Version\n"
        if version_info.get('success'):
            version_data = version_info.get('data', {})
            if isinstance(version_data, dict):
                overview += f"**Version:** {version_data.get('version', 'Unknown')}\n"
                for key, value in version_data.items():
                    if key != 'version':
                        overview += f"**{key.title()}:** {value}\n"
            else:
                overview += f"**Version Info:** {str(version_data)[:100]}...\n"
        else:
            overview += "**Version:** Unable to retrieve\n"
        
        # Performance Metrics Summary
        overview += "\n## 📈 Performance Summary\n"
        chat_test = health_result.get('chat_test', {})
        streaming_test = health_result.get('streaming_test', {})
        
        if chat_test.get('success'):
            overview += f"**Chat API Latency:** {chat_test.get('latency', 0)*1000:.0f}ms\n"
        else:
            overview += "**Chat API:** ❌ Not responding\n"
            
        if streaming_test.get('success'):
            overview += f"**Streaming Latency:** {streaming_test.get('latency', 0)*1000:.0f}ms\n"
        else:
            overview += "**Streaming:** ❌ Not available\n"
        
        # Configuration Summary
        overview += "\n## ⚙️ Configuration\n"
        overview += f"**API Endpoint:** `{self.config.api_endpoint}`\n"
        overview += f"**Default Temperature:** {self.config.default_temperature}\n"
        overview += f"**Default Max Tokens:** {self.config.default_max_tokens}\n"
        overview += f"**Auto-Stream Threshold:** {self.config.auto_stream_threshold} chars\n"
        
        return overview
    
    def get_detailed_metrics(self) -> str:
        """Get detailed metrics in formatted view (fallback for text display)"""
        metrics_result = self.client.get_metrics()
        
        if not metrics_result.get('success'):
            return f"## ❌ Metrics Unavailable\n\nError: {metrics_result.get('error', 'Unknown error')}"
        
        # Add new metrics data to collector
        raw_metrics = metrics_result.get('data', '')
        self.metrics_collector.add_metrics_data(raw_metrics)
        
        # Return summary instead of full dump
        metrics_text = "## 📊 Metrics Summary\n\n"
        
        if raw_metrics:
            parsed = self.metrics_collector.parse_prometheus_metrics(raw_metrics)
            if parsed:
                categories = defaultdict(list)
                for metric_name in parsed.keys():
                    category = self.metrics_collector.categorize_metric(metric_name)
                    categories[category].append(metric_name)
                
                for category, metrics in categories.items():
                    if metrics:
                        metrics_text += f"### {category.title()} ({len(metrics)} metrics)\n"
                        for metric in sorted(metrics)[:5]:  # Show first 5 per category
                            value = parsed.get(metric, 0)
                            metrics_text += f"- `{metric}`: {value}\n"
                        if len(metrics) > 5:
                            metrics_text += f"- ... and {len(metrics) - 5} more\n"
                        metrics_text += "\n"
        else:
            metrics_text += "No metrics data available.\n"
        
        return metrics_text
    
    def generate_sample_metrics(self) -> str:
        """Generate sample metrics data for testing when real endpoint is unavailable"""
        import random
        timestamp = datetime.now()
        
        sample_metrics = f"""# HELP memory_usage_bytes Current memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes {random.randint(1000000, 5000000)}

# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total {random.randint(100, 1000)}

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds {random.uniform(0.1, 2.0)}

# HELP prompt_tokens_total Total prompt tokens processed
# TYPE prompt_tokens_total counter
prompt_tokens_total {random.randint(10000, 100000)}

# HELP completion_tokens_total Total completion tokens generated
# TYPE completion_tokens_total counter
completion_tokens_total {random.randint(5000, 50000)}

# HELP inference_time_seconds Model inference time
# TYPE inference_time_seconds gauge
inference_time_seconds {random.uniform(0.05, 1.5)}

# HELP gpu_memory_usage GPU memory usage in bytes
# TYPE gpu_memory_usage gauge
gpu_memory_usage {random.randint(500000, 8000000)}

# HELP active_requests Currently active requests
# TYPE active_requests gauge
active_requests {random.randint(0, 50)}

# HELP tokens_per_second Token processing rate
# TYPE tokens_per_second gauge
tokens_per_second {random.uniform(10.0, 100.0)}

# HELP model_memory_usage Model memory usage in bytes
# TYPE model_memory_usage gauge
model_memory_usage {random.randint(1000000, 10000000)}
"""
        return sample_metrics

    def get_metrics_plots(self) -> Dict[str, Any]:
        """Get plotly figures for each metric category"""
        try:
            print("🔍 Starting plot generation...")
            plots = {}
            
            if not PLOTTING_AVAILABLE:
                error_msg = "Plotting not available. Install plotly to enable visual metrics."
                print(f"❌ {error_msg}")
                return {"error": error_msg}
            
            print("✅ Plotly is available")
            categories = ['memory', 'transactions', 'tokens', 'model']
            
            # Check if we have any data at all
            total_metrics = len(self.metrics_collector.metrics_data)
            total_timestamps = len(self.metrics_collector.timestamps)
            
            print(f"📊 Data check: {total_metrics} metrics, {total_timestamps} timestamps")
            
            has_data = total_metrics > 0 and total_timestamps > 0
            print(f"📊 Has existing data: {has_data}")
            
            if not has_data:
                print("🎲 Generating sample data...")
                # Add sample data for demonstration
                sample_data = self.generate_sample_metrics()
                self.metrics_collector.add_metrics_data(sample_data)
                print("✅ Sample data added")
            
            for category in categories:
                print(f"\n🔧 Processing {category} category...")
                try:
                    # First check if we have data for this category
                    cat_data = self.metrics_collector.get_metrics_by_category(category)
                    print(f"📊 Found {len(cat_data)} {category} metrics")
                    
                    fig = self.metrics_collector.create_time_series_plot(category)
                    if fig:
                        plots[category] = fig
                        print(f"✅ {category} plot created successfully")
                    else:
                        print(f"⚠️ create_time_series_plot returned None for {category}")
                        print(f"    Data check: {len(cat_data)} metrics available")
                        # Create empty placeholder plot
                        fig = go.Figure()
                        message = f"No {category} metrics available yet.<br>Start collection to see data."
                        if cat_data:
                            message = f"Found {len(cat_data)} {category} metrics but plot failed.<br>Check debug information."
                        
                        fig.add_annotation(
                            text=message,
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, xanchor='center', yanchor='middle',
                            showarrow=False,
                            font=dict(size=16, color="gray")
                        )
                        fig.update_layout(
                            title=f'{category.title()} Metrics',
                            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                            height=400
                        )
                        plots[category] = fig
                        print(f"✅ {category} placeholder created")
                except Exception as category_error:
                    print(f"❌ Error creating {category} plot: {str(category_error)}")
                    plots[category] = None
            
            print(f"🎉 Plot generation complete. Created {len([p for p in plots.values() if p is not None])} plots")
            return plots
            
        except Exception as e:
            error_msg = f"Critical error in plot generation: {str(e)}"
            print(f"💥 {error_msg}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": error_msg}
    
    def get_api_capabilities(self) -> str:
        """Get API capabilities from OpenAPI spec"""
        spec_result = self.client.get_openapi_spec()
        
        if not spec_result.get('success'):
            return f"## ❌ API Specification Unavailable\n\nError: {spec_result.get('error', 'Unknown error')}"
        
        spec = spec_result.get('data', {})
        capabilities = "## 🚀 API Capabilities\n\n"
        
        # Extract basic info
        info = spec.get('info', {})
        if info:
            capabilities += f"**API Title:** {info.get('title', 'Unknown')}\n"
            capabilities += f"**API Version:** {info.get('version', 'Unknown')}\n\n"
        
        # Extract available endpoints
        paths = spec.get('paths', {})
        if paths:
            capabilities += "### Available Endpoints:\n"
            
            # Group endpoints by category
            categorized = {
                'Core': [],
                'Models': [],
                'Health': [],
                'Utilities': [],
                'Other': []
            }
            
            for path, methods in paths.items():
                if '/v1/' in path:
                    categorized['Core'].append(path)
                elif 'model' in path.lower():
                    categorized['Models'].append(path)
                elif any(x in path.lower() for x in ['health', 'ping', 'metrics']):
                    categorized['Health'].append(path)
                elif any(x in path.lower() for x in ['tokenize', 'version', 'docs']):
                    categorized['Utilities'].append(path)
                else:
                    categorized['Other'].append(path)
            
            for category, endpoints in categorized.items():
                if endpoints:
                    capabilities += f"\n**{category}:**\n"
                    for endpoint in sorted(endpoints):
                        capabilities += f"- `{endpoint}`\n"
        
        return capabilities
    
    def run_diagnostics(self) -> str:
        """Run comprehensive diagnostics"""
        print("🔍 Running enhanced diagnostics...")
        results = self.client.test_connection()
        
        diagnostic_text = f"# 🔍 Enhanced Connection Diagnostics\n"
        diagnostic_text += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Health check
        health = results.get('health', {})
        status = "✅ PASS" if health.get('success') else "❌ FAIL"
        diagnostic_text += f"## 1. Health Check: {status}\n"
        if health.get('success'):
            diagnostic_text += f"- Latency: {health.get('latency', 0):.2f}s\n"
        else:
            diagnostic_text += f"- Error: {health.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Models check
        models = results.get('models', {})
        status = "✅ PASS" if models.get('success') else "❌ FAIL"
        diagnostic_text += f"## 2. Models Check: {status}\n"
        if models.get('success'):
            available = models.get('available', [])
            configured = models.get('configured', '')
            match = models.get('match', False)
            diagnostic_text += f"- Available: {available}\n"
            diagnostic_text += f"- Configured: {configured}\n"
            diagnostic_text += f"- Match: {'✅' if match else '❌'}\n"
            if not match and available:
                diagnostic_text += f"- **💡 Try:** {available[0]}\n"
        else:
            diagnostic_text += f"- Error: {models.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Chat test
        chat_test = results.get('chat_test', {})
        status = "✅ PASS" if chat_test.get('success') else "❌ FAIL"
        diagnostic_text += f"## 3. Chat API Test: {status}\n"
        if chat_test.get('success'):
            diagnostic_text += f"- Latency: {chat_test.get('latency', 0):.2f}s\n"
        else:
            diagnostic_text += f"- Status: {chat_test.get('status', 'Unknown')}\n"
            diagnostic_text += f"- Error: {chat_test.get('error', chat_test.get('response', 'Unknown'))}\n"
        diagnostic_text += "\n"
        
        # Streaming test
        streaming_test = results.get('streaming_test', {})
        status = "✅ PASS" if streaming_test.get('success') else "❌ FAIL"
        diagnostic_text += f"## 4. Streaming Test: {status}\n"
        if streaming_test.get('success'):
            diagnostic_text += f"- Latency: {streaming_test.get('latency', 0):.2f}s\n"
            diagnostic_text += f"- Content received: {'✅' if streaming_test.get('content_received') else '❌'}\n"
        else:
            diagnostic_text += f"- Error: {streaming_test.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Recommendations
        diagnostic_text += "## 💡 Recommendations\n"
        if not health.get('success'):
            diagnostic_text += "- Check network connectivity and API endpoint\n"
        if not models.get('match', True):
            diagnostic_text += "- Update model name in configuration\n"
        if not chat_test.get('success'):
            diagnostic_text += "- API may be overloaded, try again later\n"
        if not streaming_test.get('success'):
            diagnostic_text += "- Streaming disabled, large contexts may timeout\n"
        
        return diagnostic_text
    
    def create_interface(self) -> gr.Blocks:
        """Create enhanced Gradio interface"""
        
        # Custom CSS for better layout and readability
        custom_css = """
        /* Reset default Gradio constraints */
        main, .main, .w-full {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Make the interface use full width */
        .container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        
        /* Full width for main gradio container */
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 20px !important;
        }
        
        /* Override any max-width constraints */
        div[class*="max-w-"] {
            max-width: 100% !important;
        }
        
        /* Increase chat message font size and spacing */
        .message-wrap {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        
        /* Better spacing for chat bubbles */
        .message {
            padding: 12px 20px !important;
            margin: 10px 0 !important;
        }
        
        /* Full width chat area */
        #chatbot {
            max-width: none !important;
            width: 100% !important;
        }
        
        /* Better code block styling */
        .message pre {
            background-color: #f4f4f4 !important;
            padding: 12px !important;
            border-radius: 6px !important;
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        
        /* Improve button spacing */
        .gr-button {
            margin: 2px !important;
        }
        
        /* Configuration panel styling */
        .config-panel {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            height: 100%;
            min-width: 280px;
        }
        
        /* Tab container full width */
        .tabs {
            width: 100% !important;
        }
        
        /* Chat tab content full width */
        .tabitem {
            width: 100% !important;
        }
        
        /* Message input full width */
        .gr-text-input {
            width: 100% !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .config-panel {
                min-width: 100%;
            }
        }
        
        /* Hide duplicate progress bars */
        .progress-bar:nth-of-type(n+2) {
            display: none !important;
        }
        
        /* Metrics tab styling */
        .metrics-tab {
            padding: 10px;
        }
        
        /* Plot containers */
        .plot-container {
            margin: 10px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Metrics controls styling */
        .metrics-controls {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        /* Color coding for metric categories */
        .memory-metrics { border-left: 4px solid #FF6B6B; }
        .transaction-metrics { border-left: 4px solid #4ECDC4; }
        .token-metrics { border-left: 4px solid #45B7D1; }
        .model-metrics { border-left: 4px solid #96CEB4; }
        
        /* Limit progress bars to single instance */
        .gradio-container .wrap:has(.progress-bar) .progress-bar ~ .progress-bar {
            display: none !important;
        }
        
        /* Clean up processing indicators */
        .processing-indicator {
            max-height: 4px !important;
        }
        """
        
        with gr.Blocks(title="Telco-AIX SME Web Interface", theme=gr.themes.Soft(), css=custom_css) as interface:
            
            # Header - more compact
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown(
                        """
                        # 🤖 Telco-AIX SME Web Interface
                        **Connected to:** Qwen3-32B Model | **Features:** Smart streaming, timeout handling, context optimization
                        """
                    )
                with gr.Column(scale=1):
                    # Status indicator in header
                    if self.client.health_check():
                        gr.Markdown("✅ **Status:** Online")
                    else:
                        gr.Markdown("❌ **Status:** Offline")
            
            # Session management row
            with gr.Row():
                with gr.Column(scale=2):
                    session_id_input = gr.Textbox(
                        label="🔑 Session ID",
                        placeholder="Leave empty for new session or enter existing session ID",
                        value="",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    load_session_btn = gr.Button("📂 Load Session", variant="secondary")
                    new_session_btn = gr.Button("🆕 New Session", variant="primary")
                    list_sessions_btn = gr.Button("📋 List Sessions", variant="secondary")
            
            # Sessions list display (collapsible)
            with gr.Accordion("📂 Active Sessions", open=False) as sessions_accordion:
                with gr.Row():
                    with gr.Column(scale=3):
                        sessions_display = gr.Markdown("Click 'List Sessions' to view active sessions...")
                    with gr.Column(scale=1):
                        gr.Markdown("**Quick Load:**")
                        session_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Session",
                            interactive=True,
                            show_label=False
                        )
                        quick_load_btn = gr.Button("⚡ Quick Load", variant="primary", size="sm")
            
            # Main content area with tabs for better organization
            with gr.Tabs():
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        # Main chat column - use most of the screen
                        with gr.Column(scale=5):
                            # Chat interface
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=700,
                                elem_id="chatbot",
                                type="tuples",
                                show_label=False,
                                container=True,
                                scale=1,
                                layout="panel"
                            )
                            
                            # Message input area
                            with gr.Row():
                                msg = gr.Textbox(
                                    label="Message",
                                    placeholder="Type your message here... (Auto-streaming for large contexts)",
                                    lines=3,
                                    scale=5,
                                    show_label=False
                                )
                            
                            # Action buttons
                            with gr.Row():
                                submit = gr.Button("Send", variant="primary", scale=1)
                                clear = gr.Button("🗑️ Clear", scale=1)
                                export = gr.Button("📥 Export", scale=1)
                                file_upload = gr.File(
                                    label="📎 Attach",
                                    file_types=[".txt", ".md", ".csv", ".json", ".py"],
                                    file_count="single",
                                    scale=1
                                )
                            
                            # Context info
                            context_info = gr.Markdown(f"**Context:** Ready | **Mode:** Direct | **Temp:** {self.config.default_temperature} | **Tokens:** {self.config.default_max_tokens}", elem_id="context-info")
                            
                            # Export output (hidden by default)
                            export_output = gr.Textbox(
                                label="📄 Exported Conversation",
                                visible=False,
                                lines=10
                            )
                        
                        # Configuration sidebar - narrower
                        with gr.Column(scale=1, elem_classes="config-panel"):
                            gr.Markdown("### ⚙️ Settings")
                            
                            system_dropdown = gr.Dropdown(
                                choices=list(self.system_prompts.keys()),
                                value="Telco Expert",
                                label="System Prompt",
                                scale=1
                            )
                            
                            # Model parameters - always visible
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.config.default_temperature,
                                step=0.1,
                                label="🌡️ Temperature",
                                scale=1,
                                info="Controls randomness (0=focused, 1=creative)",
                                interactive=True
                            )
                            
                            max_tokens = gr.Slider(
                                minimum=100,
                                maximum=8192,
                                value=self.config.default_max_tokens,
                                step=100,
                                label="📏 Max Tokens",
                                scale=1,
                                info="Maximum response length",
                                interactive=True
                            )
                            
                            # Parameter status display
                            param_status = gr.Markdown(
                                f"**Current:** Temp={self.config.default_temperature} | Tokens={self.config.default_max_tokens}",
                                elem_id="param-status"
                            )
                            
                            # Custom system prompt (always accessible)
                            custom_system = gr.Textbox(
                                label="🎯 Selected System Prompt Detail",
                                placeholder="Override selected template with custom prompt...",
                                lines=4,
                                scale=1,
                                info="To edit templates permanently, use the 📝 Prompt Manager tab."
                            )
                
                # Additional tabs for better organization
                with gr.TabItem("🎛️ Management"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Overview section with auto-refresh
                            gr.Markdown("### 📊 Model Serving Overview")
                            with gr.Row():
                                refresh_overview_btn = gr.Button("🔄 Refresh Overview", variant="primary")
                                auto_refresh_checkbox = gr.Checkbox(label="Auto-refresh (30s)", value=False)
                            
                            management_overview = gr.Markdown(
                                value="Click 'Refresh Overview' to load current status...",
                                elem_id="management-overview"
                            )
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### 🛠️ Quick Actions")
                            diagnostics_btn = gr.Button("🔍 Run Full Diagnostics", variant="secondary", size="sm")
                            test_streaming_btn = gr.Button("🧪 Test Streaming", variant="secondary", size="sm")
                            test_ui_btn = gr.Button("🎯 Test UI Update", variant="secondary", size="sm")
                            
                            gr.Markdown("### 📝 Status")
                            last_update_display = gr.Textbox(
                                label="Last Update",
                                value="Not updated yet",
                                interactive=False,
                                max_lines=1
                            )
                    
                    with gr.Row():
                        # Tabbed sections for detailed information
                        with gr.Tabs():
                            with gr.TabItem("📊 Metrics"):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        # Metrics controls
                                        with gr.Row():
                                            start_collection_btn = gr.Button("▶️ Start Collection", variant="primary", size="sm")
                                            stop_collection_btn = gr.Button("⏸️ Stop Collection", variant="secondary", size="sm")
                                            sample_data_btn = gr.Button("🎲 Generate Sample Data", variant="secondary", size="sm")
                                            debug_btn = gr.Button("🔧 Debug Info", variant="secondary", size="sm")
                                            test_plots_btn = gr.Button("🧪 Test Plots", variant="secondary", size="sm")
                                            refresh_plots_btn = gr.Button("🔄 Refresh Plots", variant="primary", size="sm")
                                            force_plots_btn = gr.Button("💪 Force Plots", variant="primary", size="sm")
                                            
                                        with gr.Row():
                                            pull_interval_slider = gr.Slider(
                                                minimum=5,
                                                maximum=300,
                                                value=30,
                                                step=5,
                                                label="Pull Interval (seconds)",
                                                interactive=True
                                            )
                                        
                                        collection_status = gr.Textbox(
                                            label="Collection Status",
                                            value="Not started",
                                            interactive=False,
                                            max_lines=1
                                        )
                                
                                # Visual metrics plots organized by category
                                with gr.Row():
                                    with gr.Tabs():
                                        with gr.TabItem("🧠 Memory", elem_classes="metrics-tab"):
                                            memory_plot = gr.Plot(
                                                label="Memory Metrics Over Time"
                                            )
                                        
                                        with gr.TabItem("🔄 Transactions", elem_classes="metrics-tab"):
                                            transactions_plot = gr.Plot(
                                                label="Transaction Metrics Over Time"
                                            )
                                        
                                        with gr.TabItem("🎯 Tokens", elem_classes="metrics-tab"):
                                            tokens_plot = gr.Plot(
                                                label="Token Metrics Over Time"
                                            )
                                        
                                        with gr.TabItem("⚙️ Model", elem_classes="metrics-tab"):
                                            model_plot = gr.Plot(
                                                label="Model Performance Metrics Over Time"
                                            )
                                
                                # Archive management section
                                with gr.Accordion("💾 Data Archive", open=False):
                                    with gr.Row():
                                        export_btn = gr.Button("📤 Export Data", variant="secondary", size="sm")
                                        import_btn = gr.Button("📥 Import Data", variant="secondary", size="sm")
                                        clear_archive_btn = gr.Button("🗑️ Clear Archive", variant="stop", size="sm")
                                    
                                    with gr.Row():
                                        export_filename = gr.Textbox(
                                            label="Export Filename (optional)",
                                            placeholder="metrics_export_custom.json",
                                            scale=2
                                        )
                                        import_file_dropdown = gr.Dropdown(
                                            label="Import File",
                                            choices=[],
                                            scale=2,
                                            interactive=True
                                        )
                                        refresh_files_btn = gr.Button("🔄", size="sm", scale=1)
                                    
                                    archive_status = gr.Textbox(
                                        label="Archive Status",
                                        value="Archive operations will show status here...",
                                        interactive=False,
                                        lines=2
                                    )
                                
                                # Fallback text display
                                with gr.Accordion("📋 Text Summary", open=False):
                                    metrics_output = gr.Markdown(
                                        value="Metrics summary will appear here after collection starts...",
                                        elem_id="metrics-display"
                                    )
                            
                            with gr.TabItem("🚀 API Capabilities"):
                                capabilities_output = gr.Markdown(
                                    value="API capabilities will appear here after refresh...",
                                    elem_id="capabilities-display"
                                )
                            
                            with gr.TabItem("🔍 Diagnostics Log"):
                                diagnostics_output = gr.Textbox(
                                    label="📊 Diagnostics Report",
                                    lines=15,
                                    max_lines=25,
                                    value="Click 'Run Full Diagnostics' to test all endpoints..."
                                )
                
                with gr.TabItem("📂 Sessions"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Session Information")
                            sessions_info = gr.Textbox(
                                label="📋 Active Sessions",
                                lines=15,
                                max_lines=20,
                                value="Click 'List Sessions' to view active sessions...",
                                interactive=False
                            )
                            
                            with gr.Row():
                                refresh_sessions_btn = gr.Button("🔄 Refresh Sessions", variant="secondary")
                                cleanup_sessions_btn = gr.Button("🧹 Cleanup Expired", variant="secondary")
                
                with gr.TabItem("📝 Prompt Manager"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### System Prompt Management")
                            
                            # Existing prompts section
                            gr.Markdown("### 📚 Existing Prompts")
                            existing_prompt_dropdown = gr.Dropdown(
                                choices=list(self.system_prompts.keys()),
                                value=list(self.system_prompts.keys())[0] if self.system_prompts else None,
                                label="Select Prompt to Edit",
                                interactive=True
                            )
                            
                            load_prompt_btn = gr.Button("📖 Load Selected Prompt", variant="secondary")
                            
                            with gr.Row():
                                reload_prompts_btn = gr.Button("🔄 Reload from File", variant="secondary")
                                delete_prompt_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                            
                            prompt_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Ready to manage prompts..."
                            )
                            
                        with gr.Column(scale=2):
                            gr.Markdown("### ✏️ Edit Prompt")
                            edit_prompt_name = gr.Textbox(
                                label="Prompt Name",
                                placeholder="e.g., Security Expert, Project Manager...",
                                interactive=True
                            )
                            edit_prompt_content = gr.Textbox(
                                label="Prompt Content",
                                placeholder="Enter the full system prompt...",
                                lines=10,
                                max_lines=20,
                                interactive=True
                            )
                            
                            with gr.Row():
                                save_prompt_btn = gr.Button("💾 Save/Update Prompt", variant="primary")
                                clear_form_btn = gr.Button("🆕 Clear Form", variant="secondary")
                            
                            gr.Markdown(
                                """
                                ### 💡 Usage Tips
                                - **Load**: Select a prompt and click "Load" to edit
                                - **Save**: Updates existing or creates new prompt
                                - **Delete**: Removes selected prompt permanently
                                - **Clear**: Start fresh with empty form
                                - All changes are saved to `system_prompts.json`
                                """
                            )
            
            # Event handlers
            def update_system_prompt(selection):
                return self.system_prompts.get(selection, "")
            
            def update_context_info(history, message, temp, tokens):
                if not history:
                    return f"**Context:** Ready | **Mode:** Direct | **Temp:** {temp} | **Tokens:** {tokens}"
                
                total_chars = sum(len(h[0]) + len(h[1]) for h in history) + len(message or "")
                mode = "🌊 Streaming" if total_chars > 4000 else "⚡ Direct"
                return f"**Context:** {total_chars} chars | **Mode:** {mode} | **Temp:** {temp} | **Tokens:** {tokens}"
            
            def update_param_status(temp, tokens):
                return f"**Current:** Temp={temp} | Tokens={tokens}"
            
            # Wire up the system dropdown
            system_dropdown.change(
                update_system_prompt,
                inputs=[system_dropdown],
                outputs=[custom_system]
            )
            
            # Update context info on message change
            msg.change(
                update_context_info,
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info]
            )
            
            # Update context info and param status when sliders change
            temperature.change(
                lambda h, m, t, tok: [update_context_info(h, m, t, tok), update_param_status(t, tok)],
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info, param_status]
            )
            
            max_tokens.change(
                lambda h, m, t, tok: [update_context_info(h, m, t, tok), update_param_status(t, tok)],
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info, param_status]
            )
            
            # Management tab handlers
            def refresh_management_overview():
                """Refresh all management information"""
                try:
                    overview = self.get_management_overview()
                except Exception as e:
                    overview = f"# ❌ Error Loading Overview\n\n{str(e)}"
                
                try:
                    metrics = self.get_detailed_metrics()
                except Exception as e:
                    metrics = f"## ❌ Error Loading Metrics\n\n{str(e)}"
                
                try:
                    capabilities = self.get_api_capabilities()
                except Exception as e:
                    capabilities = f"## ❌ Error Loading Capabilities\n\n{str(e)}"
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return overview, metrics, capabilities, timestamp
            
            # Metrics collection handlers
            def start_metrics_collection(interval):
                """Start metrics collection with specified interval"""
                try:
                    self.metrics_collector.set_pull_interval(int(interval))
                    self.metrics_collector.start_collection(self.client)
                    return "✅ Collection started"
                except Exception as e:
                    return f"❌ Error starting collection: {str(e)}"
            
            def stop_metrics_collection():
                """Stop metrics collection"""
                try:
                    self.metrics_collector.stop_collection()
                    return "⏸️ Collection stopped"
                except Exception as e:
                    return f"❌ Error stopping collection: {str(e)}"
            
            def update_pull_interval(interval):
                """Update the pull interval for metrics collection"""
                try:
                    self.metrics_collector.set_pull_interval(int(interval))
                    return f"⏱️ Interval updated to {interval}s"
                except Exception as e:
                    return f"❌ Error updating interval: {str(e)}"
            
            def refresh_metrics_plots():
                """Refresh all metrics plots"""
                try:
                    print("🔄 Refreshing metrics plots...")
                    plots = self.get_metrics_plots()
                    
                    if "error" in plots:
                        error_msg = plots["error"]
                        print(f"❌ Plot error: {error_msg}")
                        return None, None, None, None, error_msg
                    
                    print(f"📊 Retrieved plots: {list(plots.keys())}")
                    
                    memory_fig = plots.get('memory')
                    transactions_fig = plots.get('transactions')
                    tokens_fig = plots.get('tokens')
                    model_fig = plots.get('model')
                    
                    print(f"🧠 Memory plot: {'✅ Present' if memory_fig else '❌ None'}")
                    print(f"🔄 Transactions plot: {'✅ Present' if transactions_fig else '❌ None'}")
                    print(f"🎯 Tokens plot: {'✅ Present' if tokens_fig else '❌ None'}")
                    print(f"⚙️ Model plot: {'✅ Present' if model_fig else '❌ None'}")
                    
                    metrics_summary = self.get_detailed_metrics()
                    
                    return (
                        memory_fig,
                        transactions_fig, 
                        tokens_fig,
                        model_fig,
                        metrics_summary
                    )
                except Exception as e:
                    print(f"💥 Error in refresh_metrics_plots: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return None, None, None, None, f"Error refreshing plots: {str(e)}"
            
            def generate_sample_data():
                """Generate and add sample metrics data"""
                try:
                    print("🎲 Generating sample data...")
                    sample_data = self.generate_sample_metrics()
                    self.metrics_collector.add_metrics_data(sample_data)
                    
                    # Get updated plots
                    plots = self.get_metrics_plots()
                    
                    if "error" in plots:
                        error_msg = plots["error"]
                        print(f"❌ Plot error: {error_msg}")
                        
                        # Create simple text-based fallback
                        fallback_msg = f"📊 Sample data added (Text mode)\n\n{error_msg}"
                        return None, None, None, None, fallback_msg, self.get_detailed_metrics()
                    
                    print("✅ Sample data and plots generated successfully")
                    return (
                        plots.get('memory'),
                        plots.get('transactions'), 
                        plots.get('tokens'),
                        plots.get('model'),
                        "📊 Sample data generated successfully",
                        self.get_detailed_metrics()
                    )
                except Exception as e:
                    print(f"💥 Critical error: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return None, None, None, None, f"❌ Error generating sample data: {str(e)}", str(e)
            
            def force_create_plots():
                """Force create plots from any available metrics data"""
                try:
                    if not PLOTTING_AVAILABLE:
                        return None, None, None, None, "❌ Plotly not available"
                    
                    import plotly.graph_objects as go
                    
                    # Get all metrics regardless of category
                    all_metrics = dict(self.metrics_collector.metrics_data)
                    timestamps = list(self.metrics_collector.timestamps)
                    
                    if not all_metrics:
                        return None, None, None, None, "❌ No metrics data found"
                    
                    # Split metrics into 4 groups for the 4 plot areas
                    metric_names = list(all_metrics.keys())
                    
                    # Create 4 figures with whatever metrics we have
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green
                    titles = ['Memory', 'Transactions', 'Tokens', 'Model']
                    
                    figures = []
                    
                    for i in range(4):
                        fig = go.Figure()
                        
                        # Take every 4th metric starting from i
                        group_metrics = metric_names[i::4]
                        
                        for j, metric_name in enumerate(group_metrics[:5]):  # Max 5 per plot
                            values = list(all_metrics[metric_name])
                            if values:
                                # Create timestamps if needed
                                if len(timestamps) != len(values):
                                    base_time = datetime.now()
                                    plot_timestamps = [base_time - timedelta(seconds=(len(values)-1-k)*30) for k in range(len(values))]
                                else:
                                    plot_timestamps = timestamps[-len(values):]
                                
                                fig.add_trace(go.Scatter(
                                    x=plot_timestamps,
                                    y=values,
                                    mode='lines+markers',
                                    name=metric_name,
                                    line=dict(color=colors[i], width=2),
                                    marker=dict(size=4)
                                ))
                        
                        fig.update_layout(
                            title=f'{titles[i]} Metrics ({len(group_metrics)} total)',
                            xaxis_title='Time',
                            yaxis_title='Value',
                            height=400
                        )
                        
                        figures.append(fig if fig.data else None)
                    
                    return figures[0], figures[1], figures[2], figures[3], f"✅ Created plots from {len(all_metrics)} metrics"
                    
                except Exception as e:
                    print(f"❌ Force create error: {str(e)}")
                    return None, None, None, None, f"❌ Error: {str(e)}"
            
            def debug_info():
                """Get debug information about the plotting system"""
                try:
                    debug_msg = "🔧 Debug Information\n\n"
                    
                    # Check Plotly availability
                    debug_msg += f"**Plotly Available:** {PLOTTING_AVAILABLE}\n"
                    
                    if PLOTTING_AVAILABLE:
                        try:
                            import plotly
                            debug_msg += f"**Plotly Version:** {plotly.__version__}\n"
                        except:
                            debug_msg += "**Plotly Version:** Could not determine\n"
                    
                    # Check metrics data
                    total_metrics = len(self.metrics_collector.metrics_data)
                    debug_msg += f"**Total Metrics Collected:** {total_metrics}\n"
                    debug_msg += f"**Timestamps Available:** {len(self.metrics_collector.timestamps)}\n"
                    
                    # Show actual metric names that were collected
                    debug_msg += "\n**📊 Collected Metric Names:**\n"
                    metric_names = list(self.metrics_collector.metrics_data.keys())
                    for i, name in enumerate(metric_names[:20]):  # Show first 20
                        category = self.metrics_collector.categorize_metric(name)
                        debug_msg += f"- `{name}` → {category}\n"
                    if len(metric_names) > 20:
                        debug_msg += f"- ... and {len(metric_names) - 20} more\n"
                    
                    # Check categories
                    debug_msg += "\n**📈 Category Breakdown:**\n"
                    for category in ['memory', 'transactions', 'tokens', 'model']:
                        cat_data = self.metrics_collector.get_metrics_by_category(category)
                        debug_msg += f"**{category.title()}:** {len(cat_data)} metrics\n"
                        for metric_name in list(cat_data.keys())[:3]:  # Show first 3 per category
                            debug_msg += f"  - {metric_name}\n"
                    
                    # Count uncategorized metrics
                    uncategorized_count = 0
                    uncategorized_names = []
                    for name in metric_names:
                        if self.metrics_collector.categorize_metric(name) == 'other':
                            uncategorized_count += 1
                            if len(uncategorized_names) < 5:
                                uncategorized_names.append(name)
                    
                    debug_msg += f"\n**❓ Uncategorized:** {uncategorized_count} metrics\n"
                    for name in uncategorized_names:
                        debug_msg += f"  - {name}\n"
                    
                    # Try creating a simple plot
                    if PLOTTING_AVAILABLE:
                        try:
                            import plotly.graph_objects as go
                            test_fig = go.Figure()
                            test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], name='Test'))
                            debug_msg += "\n**Test Plot Creation:** ✅ Success\n"
                        except Exception as plot_error:
                            debug_msg += f"\n**Test Plot Creation:** ❌ Failed - {str(plot_error)}\n"
                    
                    # Add detailed plot debugging
                    debug_msg += "\n\n**🎯 PLOT DEBUGGING:**\n"
                    try:
                        for category in ['memory', 'transactions', 'tokens', 'model']:
                            debug_msg += f"\n**{category.title()} Plot Debug:**\n"
                            cat_data = self.metrics_collector.get_metrics_by_category(category)
                            debug_msg += f"  - Found {len(cat_data)} categorized metrics\n"
                            if cat_data:
                                for metric_name, data in list(cat_data.items())[:2]:  # Show first 2
                                    values = data.get('values', [])
                                    timestamps = data.get('timestamps', [])
                                    debug_msg += f"  - {metric_name}: {len(values)} values, {len(timestamps)} timestamps\n"
                                
                                # Try creating the plot
                                try:
                                    fig = self.metrics_collector.create_time_series_plot(category)
                                    if fig:
                                        debug_msg += f"  - Plot creation: ✅ SUCCESS\n"
                                    else:
                                        debug_msg += f"  - Plot creation: ❌ RETURNED NONE\n"
                                except Exception as plot_err:
                                    debug_msg += f"  - Plot creation: ❌ ERROR: {str(plot_err)}\n"
                            else:
                                debug_msg += f"  - No categorized data found\n"
                    except Exception as plot_debug_err:
                        debug_msg += f"❌ Plot debug error: {str(plot_debug_err)}\n"
                    
                    return debug_msg
                    
                except Exception as e:
                    return f"❌ Debug error: {str(e)}"
            
            # Archive management handlers
            def export_data(filename):
                """Export metrics data"""
                try:
                    result = self.metrics_collector.export_metrics(filename if filename.strip() else None)
                    return result
                except Exception as e:
                    return f"❌ Export error: {str(e)}"
            
            def import_data(selected_file):
                """Import metrics data"""
                try:
                    if not selected_file:
                        return "❌ Please select a file to import"
                    result = self.metrics_collector.import_metrics(selected_file)
                    # Refresh plots after import
                    plots = self.get_metrics_plots()
                    return result
                except Exception as e:
                    return f"❌ Import error: {str(e)}"
            
            def clear_archive():
                """Clear archived data"""
                try:
                    with self.metrics_collector.lock:
                        self.metrics_collector.metrics_data.clear()
                        self.metrics_collector.timestamps.clear()
                    
                    # Remove archive file
                    if self.metrics_collector.archive_file.exists():
                        self.metrics_collector.archive_file.unlink()
                    
                    return "✅ Archive cleared successfully"
                except Exception as e:
                    return f"❌ Clear error: {str(e)}"
            
            def refresh_import_files():
                """Refresh list of available import files"""
                try:
                    archive_dir = self.metrics_collector.archive_dir
                    json_files = []
                    
                    if archive_dir.exists():
                        # Get all JSON files in archive directory
                        for file_path in archive_dir.glob("*.json"):
                            if file_path.name != "metrics_data.json":  # Skip current archive
                                json_files.append(file_path.name)
                    
                    return gr.update(choices=sorted(json_files))
                except Exception as e:
                    print(f"❌ Error refreshing files: {str(e)}")
                    return gr.update(choices=[])
            
            def test_simple_plot():
                """Create a simple test plot to verify Gradio Plot component works"""
                try:
                    if not PLOTTING_AVAILABLE:
                        return None, None, None, None, "❌ Plotly not available"
                    
                    # Create very simple test plots
                    import plotly.graph_objects as go
                    import random
                    
                    # Simple test data
                    x = list(range(10))
                    
                    # Memory test plot (red)
                    memory_fig = go.Figure()
                    memory_fig.add_trace(go.Scatter(x=x, y=[random.randint(1000, 5000) for _ in range(10)], 
                                                   mode='lines+markers', name='memory_test',
                                                   line=dict(color='#FF6B6B', width=2)))
                    memory_fig.update_layout(title='Memory Test', height=300)
                    
                    # Transaction test plot (teal)  
                    trans_fig = go.Figure()
                    trans_fig.add_trace(go.Scatter(x=x, y=[random.randint(100, 1000) for _ in range(10)],
                                                  mode='lines+markers', name='requests_test',
                                                  line=dict(color='#4ECDC4', width=2)))
                    trans_fig.update_layout(title='Transaction Test', height=300)
                    
                    # Token test plot (blue)
                    token_fig = go.Figure()
                    token_fig.add_trace(go.Scatter(x=x, y=[random.randint(1000, 10000) for _ in range(10)],
                                                  mode='lines+markers', name='tokens_test', 
                                                  line=dict(color='#45B7D1', width=2)))
                    token_fig.update_layout(title='Token Test', height=300)
                    
                    # Model test plot (green)
                    model_fig = go.Figure()
                    model_fig.add_trace(go.Scatter(x=x, y=[random.uniform(0.1, 2.0) for _ in range(10)],
                                                  mode='lines+markers', name='inference_test',
                                                  line=dict(color='#96CEB4', width=2)))
                    model_fig.update_layout(title='Model Test', height=300)
                    
                    return memory_fig, trans_fig, token_fig, model_fig, "✅ Simple test plots created"
                    
                except Exception as e:
                    print(f"❌ Test plot error: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return None, None, None, None, f"❌ Test plot failed: {str(e)}"
            
            refresh_overview_btn.click(
                refresh_management_overview,
                outputs=[management_overview, metrics_output, capabilities_output, last_update_display]
            )
            
            # Wire up metrics collection controls
            start_collection_btn.click(
                start_metrics_collection,
                inputs=[pull_interval_slider],
                outputs=[collection_status]
            )
            
            stop_collection_btn.click(
                stop_metrics_collection,
                outputs=[collection_status]
            )
            
            pull_interval_slider.change(
                update_pull_interval,
                inputs=[pull_interval_slider],
                outputs=[collection_status]
            )
            
            # Sample data generation
            sample_data_btn.click(
                generate_sample_data,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, collection_status, metrics_output]
            )
            
            # Debug information
            debug_btn.click(
                debug_info,
                outputs=[collection_status]
            )
            
            # Test plots
            test_plots_btn.click(
                test_simple_plot,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, collection_status]
            )
            
            # Manual refresh plots
            refresh_plots_btn.click(
                refresh_metrics_plots,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, metrics_output]
            )
            
            # Force create plots from any available data
            force_plots_btn.click(
                force_create_plots,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, collection_status]
            )
            
            # Archive management handlers
            export_btn.click(
                export_data,
                inputs=[export_filename],
                outputs=[archive_status]
            )
            
            import_btn.click(
                import_data,
                inputs=[import_file_dropdown],
                outputs=[archive_status]
            )
            
            clear_archive_btn.click(
                clear_archive,
                outputs=[archive_status]
            )
            
            refresh_files_btn.click(
                refresh_import_files,
                outputs=[import_file_dropdown]
            )
            
            # Auto-refresh plots every 15 seconds when collection is active
            plots_timer = gr.Timer(15, active=False)
            plots_timer.tick(
                refresh_metrics_plots,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, metrics_output]
            )
            
            # Auto-start collection on interface load
            def auto_start_collection():
                """Auto-start metrics collection when interface loads"""
                try:
                    # Check if archive was loaded
                    archived_metrics = len(self.metrics_collector.metrics_data)
                    archived_timestamps = len(self.metrics_collector.timestamps)
                    
                    archive_msg = ""
                    if archived_metrics > 0:
                        archive_msg = f" (📦 Loaded {archived_metrics} archived metrics)"
                    
                    self.metrics_collector.set_pull_interval(30)  # 30 second default
                    self.metrics_collector.start_collection(self.client)
                    
                    # Generate initial plots
                    plots = self.get_metrics_plots()
                    
                    status_message = f"⚡ Auto-started metrics collection{archive_msg}"
                    
                    if "error" in plots:
                        return None, None, None, None, f"{status_message} (plotting disabled)", plots["error"], f"Archive loaded: {archived_metrics} metrics, {archived_timestamps} timestamps"
                    
                    return (
                        plots.get('memory'),
                        plots.get('transactions'), 
                        plots.get('tokens'),
                        plots.get('model'),
                        status_message,
                        self.get_detailed_metrics(),
                        f"Archive loaded: {archived_metrics} metrics, {archived_timestamps} timestamps"
                    )
                except Exception as e:
                    return None, None, None, None, f"❌ Auto-start failed: {str(e)}", str(e), "Archive load failed"
            
            # Initialize plots and start collection on startup
            interface.load(
                auto_start_collection,
                outputs=[memory_plot, transactions_plot, tokens_plot, model_plot, collection_status, metrics_output, archive_status]
            )
            
            # Auto-refresh functionality with timer
            timer = gr.Timer(30, active=False)  # 30 second timer, inactive by default
            
            # Auto-refresh handler
            timer.tick(
                refresh_management_overview,
                outputs=[management_overview, metrics_output, capabilities_output, last_update_display]
            )
            
            # Control timer based on checkbox
            auto_refresh_checkbox.change(
                lambda x: gr.Timer.update(active=x),
                inputs=[auto_refresh_checkbox],
                outputs=[timer]
            )
            
            # Diagnostics handlers
            diagnostics_btn.click(
                self.run_diagnostics,
                outputs=[diagnostics_output]
            )
            
            test_streaming_btn.click(
                self.test_simple_streaming,
                outputs=[diagnostics_output]
            )
            
            test_ui_btn.click(
                self.test_ui_update,
                inputs=[chatbot],
                outputs=[msg, chatbot, file_upload]
            )
            
            # Message handling with session management
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot, system_dropdown, custom_system, 
                       temperature, max_tokens, file_upload, session_id_input],
                outputs=[msg, chatbot, file_upload, session_id_input],
                show_progress="minimal"
            )
            
            submit.click(
                self.process_message,
                inputs=[msg, chatbot, system_dropdown, custom_system,
                       temperature, max_tokens, file_upload, session_id_input],
                outputs=[msg, chatbot, file_upload, session_id_input],
                show_progress="minimal"
            )
            
            # Clear history
            clear.click(
                self.clear_history,
                outputs=[chatbot]
            )
            
            # Export conversation
            def handle_export(history):
                export_text = self.export_conversation(history)
                return export_text, gr.update(visible=True)
            
            export.click(
                handle_export,
                inputs=[chatbot],
                outputs=[export_output, export_output]
            )
            
            # Prompt management handlers
            def reload_prompts_handler():
                result = self.reload_system_prompts()
                # Update all dropdown choices
                choices = list(self.system_prompts.keys())
                return (
                    result, 
                    gr.update(choices=choices),
                    gr.update(choices=choices, value=choices[0] if choices else None)
                )
            
            reload_prompts_btn.click(
                reload_prompts_handler,
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown]
            )
            
            # Load prompt for editing
            def load_prompt_handler(prompt_name):
                name, content = self.load_prompt_for_editing(prompt_name)
                status = f"Loaded prompt '{prompt_name}' for editing" if name else "No prompt selected"
                return name, content, status
            
            load_prompt_btn.click(
                load_prompt_handler,
                inputs=[existing_prompt_dropdown],
                outputs=[edit_prompt_name, edit_prompt_content, prompt_status]
            )
            
            # Save/update prompt
            def save_prompt_handler(name, content):
                result = self.save_custom_prompt(name, content)
                # Update dropdown choices if successful
                if "✅" in result:
                    choices = list(self.system_prompts.keys())
                    return (
                        result, 
                        gr.update(choices=choices),
                        gr.update(choices=choices, value=name),
                        name,  # Keep name
                        content  # Keep content
                    )
                return result, gr.update(), gr.update(), name, content
            
            save_prompt_btn.click(
                save_prompt_handler,
                inputs=[edit_prompt_name, edit_prompt_content],
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown, edit_prompt_name, edit_prompt_content]
            )
            
            # Delete prompt
            def delete_prompt_handler(prompt_name):
                result = self.delete_prompt(prompt_name)
                if "✅" in result:
                    choices = list(self.system_prompts.keys())
                    return (
                        result,
                        gr.update(choices=choices),
                        gr.update(choices=choices, value=choices[0] if choices else None),
                        "",  # Clear edit form
                        ""   # Clear edit form
                    )
                return result, gr.update(), gr.update(), gr.update(), gr.update()
            
            delete_prompt_btn.click(
                delete_prompt_handler,
                inputs=[existing_prompt_dropdown],
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown, edit_prompt_name, edit_prompt_content]
            )
            
            # Clear form
            def clear_form_handler():
                return "", "", "Form cleared - ready for new prompt"
            
            clear_form_btn.click(
                clear_form_handler,
                outputs=[edit_prompt_name, edit_prompt_content, prompt_status]
            )
            
            # Session management handlers
            def load_session_handler(session_id):
                if not session_id.strip():
                    return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens, "Please enter a session ID"
                
                history, sys_prompt, custom_prompt, temp, tokens = self.load_session(session_id)
                status = f"Loaded session {session_id} with {len(history)} messages"
                return history, sys_prompt, custom_prompt, temp, tokens, status
            
            def new_session_handler():
                history, session_id = self.new_session()
                return history, session_id, f"Created new session: {session_id}"
            
            def list_sessions_handler():
                sessions_text, session_ids = self.get_session_list()
                # Create dropdown choices with session info
                dropdown_choices = []
                sessions = self.session_manager.list_sessions()
                for session in sessions[:10]:
                    created = datetime.fromtimestamp(session['created']).strftime('%m-%d %H:%M')
                    label = f"{session['id']} ({session['messages']} msgs, {created})"
                    dropdown_choices.append((label, session['id']))
                
                # Return for displays, dropdown update, and open accordion
                return (
                    sessions_text, 
                    sessions_text, 
                    gr.update(open=True),
                    gr.update(choices=dropdown_choices)
                )
            
            def cleanup_sessions_handler():
                cleaned = self.session_manager.cleanup_expired_sessions()
                return f"Cleaned up {cleaned} expired sessions"
            
            def quick_load_handler(selected_session_id):
                if not selected_session_id:
                    return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens, selected_session_id, "Please select a session from the dropdown"
                
                history, sys_prompt, custom_prompt, temp, tokens = self.load_session(selected_session_id)
                status = f"✅ Quick loaded session {selected_session_id} ({len(history)} messages)"
                return history, sys_prompt, custom_prompt, temp, tokens, selected_session_id, status
            
            # Wire up session management
            load_session_btn.click(
                load_session_handler,
                inputs=[session_id_input],
                outputs=[chatbot, system_dropdown, custom_system, temperature, max_tokens, context_info]
            )
            
            new_session_btn.click(
                new_session_handler,
                outputs=[chatbot, session_id_input, context_info]
            )
            
            list_sessions_btn.click(
                list_sessions_handler,
                outputs=[sessions_display, sessions_info, sessions_accordion, session_dropdown]
            )
            
            refresh_sessions_btn.click(
                list_sessions_handler,
                outputs=[sessions_display, sessions_info, sessions_accordion, session_dropdown]
            )
            
            cleanup_sessions_btn.click(
                cleanup_sessions_handler,
                outputs=[sessions_info]
            )
            
            # Quick load handler
            quick_load_btn.click(
                quick_load_handler,
                inputs=[session_dropdown],
                outputs=[chatbot, system_dropdown, custom_system, temperature, max_tokens, session_id_input, context_info]
            )
        
        return interface

def main():
    """Launch the enhanced chat interface"""
    print("🚀 Starting Telco-AIX SME Web Interface...")
    
    config = Config()
    chat = ChatInterface(config)
    interface = chat.create_interface()
    
    print(f"🌐 Launching on port {30180}...")
    interface.launch(
        auth=(config.admin_username, config.admin_password),
        server_name="0.0.0.0",
        server_port=30180,
        share=False,
        inbrowser=False,
        favicon_path=None,
        ssl_verify=False,
        show_api=False,  # Hide API documentation
        max_threads=40,  # Better concurrency
        quiet=False
    )

if __name__ == "__main__":
    main()
