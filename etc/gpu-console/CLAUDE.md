# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Console is a multi-cluster GPU monitoring dashboard that provides real-time GPU metrics visualization for Kubernetes clusters with NVIDIA GPU Operator. The system consists of a React frontend with TypeScript, Node.js backend with Socket.io for real-time data, and integrates with Kubernetes clusters to execute nvidia-smi commands for GPU data collection.

## Development Commands

### Installation and Setup
```bash
# Install all dependencies (root, backend, frontend)
npm run install:all

# Start both servers in development mode
npm run dev
# or
./run-servers.sh

# Production build
npm run build

# Production start
npm start
```

### Backend Development (in backend/ directory)
```bash
# Development with hot reload
npm run dev

# Build TypeScript to JavaScript
npm run build

# Start compiled production server
npm start
```

### Frontend Development (in frontend/ directory)
```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm preview
```

### Type Checking and Linting
```bash
# Backend: TypeScript compilation check
cd backend && npx tsc --noEmit

# Frontend: TypeScript compilation check
cd frontend && npx tsc --noEmit

# Frontend: ESLint check
cd frontend && npx eslint src --ext ts,tsx
```

## Architecture Overview

### System Components
- **Backend**: Express.js server with Socket.io for real-time communication
- **Frontend**: React SPA with Chart.js for visualizations
- **Kubernetes Integration**: Uses @kubernetes/client-node to execute nvidia-smi in GPU pods
- **Data Flow**: HTTP polling every 5 seconds + WebSocket for real-time updates
- **Storage**: Browser localStorage with TTL-based cleanup

### Key Service Patterns
- **KubernetesService**: Manages kubeconfig loading and kubectl exec operations
- **Controllers**: Express route handlers following REST patterns
- **Custom Hooks**: React hooks for data fetching (useGPUData, useMultiClusterGPUData)
- **WebSocket Context**: React context for real-time data subscriptions

### Data Flow Architecture
1. Backend loads cluster configs from `config/clusters.json` on startup
2. KubernetesService finds nvidia-driver-daemonset pods in each cluster
3. Executes nvidia-smi commands via kubectl exec in pod containers
4. nvidiaSmiParser processes raw output into structured JSON
5. Frontend polls REST endpoints and subscribes to WebSocket updates
6. Metrics are persisted in browser localStorage with configurable TTL

## Important Configuration

### Cluster Configuration
Edit `config/clusters.json` to add/remove clusters:
```json
[
  {
    "name": "cluster-name",
    "kubeconfigPath": "/absolute/path/to/kubeconfig", 
    "namespace": "nvidia-gpu-operator"
  }
]
```

### Environment Variables
- `PORT`: Backend server port (default: 3001)
- `NODE_ENV`: Environment mode (development/production)
- `VITE_API_BASE_URL`: Frontend API base URL (default: http://localhost:3001)

## Development Patterns

### TypeScript Configuration
- Backend uses CommonJS modules with strict mode enabled
- Frontend uses ESNext modules with React JSX transform
- Both enforce strict type checking with additional linting rules

### Component Architecture
- Functional components with hooks pattern
- Custom hooks for data fetching and state management
- Context providers for global state (WebSocket connections)
- Chart.js integration via react-chartjs-2 wrapper components

### Error Handling
- Graceful cluster connection failures
- Per-cluster error tracking in multi-cluster mode
- Frontend error boundaries for component failures
- Socket.io reconnection handling

### Testing and GPU Data
The system executes `nvidia-smi` commands in nvidia-driver-daemonset pods within the nvidia-gpu-operator namespace. Ensure clusters have:
- NVIDIA GPU Operator installed
- nvidia-driver-daemonset pods running
- Proper RBAC permissions for pod exec operations

### Performance Considerations
- Limit to <10 clusters for optimal performance
- Configurable metrics retention (5-1440 minutes)
- Automatic localStorage cleanup with TTL
- Debounced data fetching to prevent API overload