# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python sme-web-ui.py

# Access the web interface
# URL: http://localhost:30180
# Username: admin
# Password: minad
```

### Testing and Diagnostics
The application includes built-in diagnostic capabilities accessible via the web interface:
- Navigate to "🔧 Diagnostics" tab
- Click "🔍 Run Diagnostics" to test API connections and streaming functionality
- Use "🧪 Test Streaming" button for streaming response validation

## Architecture Overview

### Core Application Structure
This is a single-file Gradio-based web application (`sme-web-ui.py`) that provides an AI chat interface for telecommunications experts. The architecture follows a modular class-based design:

#### Key Components

**SessionManager (sme-web-ui.py:94)**
- File-based session persistence using pickle
- 24-hour session retention with automatic cleanup
- Session storage in `sessions/` directory (auto-created)
- Handles session creation, loading, saving, and cleanup

**MetricsCollector (sme-web-ui.py:236)**
- Real-time metrics tracking and visualization
- Categories: Memory, Transactions, Tokens, Model performance
- Time-series data collection with Plotly integration
- Automatic metrics categorization and data aggregation

**ChatClient (sme-web-ui.py:684)**
- HTTP client for AI model API communication
- Streaming and non-streaming completion support
- Built-in retry logic with exponential backoff
- Connection testing and health check capabilities
- OpenAPI specification retrieval

**ChatInterface (sme-web-ui.py:1185)**
- Main UI orchestration and event handling
- Multi-tab interface (Chat, Prompt Manager, Diagnostics, Metrics)
- File upload processing (.txt, .md, .csv, .json, .py)
- Real-time settings management

### Configuration System
The `Config` dataclass (sme-web-ui.py:41) contains all application settings:
- API endpoint: `https://qwen3-32b-vllm-latest-tme-aix.apps.sandbox01.narlabs.io`
- Model: `qwen3-32b-vllm-latest`
- Timeouts: 45s connect, 240s read, 600s streaming
- Auto-streaming threshold: 4000 characters
- Authentication: admin/minad

### System Prompts Architecture
Expert personas are loaded from `system_prompts.json` with 5 specialized domains:
- Default Assistant (precision-focused business AI)
- Network Expert (enterprise/SP network architecture)
- Telco Expert (5G/6G, RAN, Core networks)
- Storage Expert (petabyte-scale storage design)
- Cloud Expert (multi-cloud transformations)

Each prompt contains detailed technical expertise, vendor knowledge, methodologies, and deliverable frameworks.

### Data Flow
1. **Session Management**: Sessions persist across browser refreshes and server restarts
2. **Message Processing**: Context size management with auto-streaming for large contexts
3. **File Processing**: 3,500 character limit, one-time use per message
4. **Metrics Collection**: Real-time performance tracking across all interactions
5. **Response Handling**: Dual-mode (streaming/non-streaming) with automatic fallback

### Key Technical Features
- **Smart Context Management**: Automatic streaming activation based on context size
- **Thread Safety**: Concurrent request handling with processing locks
- **Retry Mechanisms**: Robust error handling with exponential backoff
- **Real-time Metrics**: Live performance monitoring with Plotly visualizations
- **Session Persistence**: Survives application restarts and browser refreshes

## File Structure
```
├── sme-web-ui.py           # Main application (single file architecture)
├── system_prompts.json     # Expert persona definitions
├── sessions/               # Session storage (auto-created)
├── benchmarks/             # AI model benchmark test files
├── requirements.txt        # Python dependencies (gradio, plotly)
└── README.md              # User documentation
```

## Development Notes

### Model Integration
The application is designed for the Qwen3-32B model running on RHOAI ModelServing with vLLM runtime and NVIDIA acceleration. The integration includes:
- OpenAI-compatible API endpoints
- Streaming and non-streaming response handling
- Model-specific parameter tuning (temperature, max_tokens)
- SSL verification disabled for development environments

### Session Management Implementation
Sessions use an 8-character unique identifier system with pickle-based persistence. The implementation includes automatic cleanup, concurrent access handling, and recovery mechanisms for corrupted session files.

### Metrics System
The metrics collection system categorizes data into Memory, Transactions, Tokens, and Model performance metrics with timestamp alignment and real-time visualization capabilities using Plotly.

### UI Framework
Built on Gradio with custom CSS styling, the interface provides tabbed navigation, real-time updates, drag-and-drop file uploads, and responsive design elements optimized for professional telecommunications use cases.