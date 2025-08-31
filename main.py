"""
IoT Node Placement Simulator - FastAPI Backend
Optimizing Gateway-Based IoT Node Placement using Differential Evolution
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import io
import base64
import json

app = FastAPI(
    title="IoT Node Placement Simulator",
    description="Optimizing Gateway-Based IoT Node Placement using Differential Evolution",
    version="2.0.0"
)

# Configure CORS
allowed_origins = [
    "http://localhost:3000", 
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001"
]

# Add production frontend URL from environment variable if available
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class OptimizationRequest(BaseModel):
    field_size: float = 80000
    n_sensors: int = 10
    k_gateways: int = 2
    alpha: float = 1.0
    beta: float = 10.0

class OptimizationResponse(BaseModel):
    fitness: float
    gateways: List[List[float]]
    load_distribution: List[int]
    plot: str
    convergence_info: Dict[str, Any]
    sensor_positions: List[List[float]]
    master_node: List[float]

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for caching
_cached_sensors = None
_cached_field_size = None

def generate_sensors(n_sensors: int, field_size: float) -> np.ndarray:
    """Generate random sensor positions in the field"""
    global _cached_sensors, _cached_field_size
    
    # Cache sensors for consistent results during optimization
    if _cached_sensors is None or _cached_field_size != field_size:
        np.random.seed(42)
        _cached_sensors = np.random.uniform(0, field_size, size=(n_sensors, 2))
        _cached_field_size = field_size
    
    return _cached_sensors[:n_sensors]

def compute_fitness(gateway_coords_flat: np.ndarray, sensor_positions: np.ndarray, 
                   master_node: np.ndarray, k_gateways: int, alpha: float, beta: float) -> float:
    """
    Compute fitness score for gateway placement
    Lower fitness = better placement
    """
    gateways = gateway_coords_flat.reshape((k_gateways, 2))
    total_distance = 0.0
    load = np.zeros(k_gateways)
    
    # Assign each sensor to nearest gateway
    for sensor in sensor_positions:
        dists = np.linalg.norm(gateways - sensor, axis=1)
        nearest_idx = np.argmin(dists)
        total_distance += dists[nearest_idx]
        load[nearest_idx] += 1
    
    # Calculate load variance (penalty for uneven distribution)
    load_variance = np.var(load)
    
    # Combined fitness function
    fitness = alpha * total_distance + beta * load_variance
    
    return fitness

def create_network_visualization(gateways: np.ndarray, sensor_positions: np.ndarray, 
                                master_node: np.ndarray, field_size: float) -> str:
    """Create and return base64 encoded network visualization"""
    
    # Calculate sensor-to-gateway assignments
    sensor_to_gateway = []
    for sensor in sensor_positions:
        dists = np.linalg.norm(gateways - sensor, axis=1)
        nearest_idx = np.argmin(dists)
        sensor_to_gateway.append(nearest_idx)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot sensors with different colors for each gateway assignment
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    for i, sensor in enumerate(sensor_positions):
        gateway_idx = sensor_to_gateway[i]
        color = colors[gateway_idx % len(colors)]
        plt.scatter(sensor[0], sensor[1], color=color, s=100, alpha=0.8, 
                   edgecolors='blue', linewidth=2, label=f'Gateway {gateway_idx + 1} Sensors' if i == 0 else "")
    
    # Plot gateways
    gateway_colors = ['red', 'darkred', 'maroon', 'brown', 'crimson', 'darkslategray']
    for i, gateway in enumerate(gateways):
        color = gateway_colors[i % len(gateway_colors)]
        plt.scatter(gateway[0], gateway[1], color=color, marker='s', s=250, 
                   label=f'Gateway {i + 1}', alpha=0.9, edgecolors='black', linewidth=2)
    
    # Plot master node
    plt.scatter(master_node[0], master_node[1], color='black', marker='*', 
               s=400, label='Master/Cloud', alpha=1.0, edgecolors='white', linewidth=2)
    
    # Draw connections from sensors to gateways
    for i, sensor in enumerate(sensor_positions):
        gateway = gateways[sensor_to_gateway[i]]
        plt.plot([sensor[0], gateway[0]], [sensor[1], gateway[1]], 
                'gray', linestyle='--', linewidth=1, alpha=0.6)
    
    # Draw connections from gateways to master node
    for gateway in gateways:
        plt.plot([gateway[0], master_node[0]], [gateway[1], master_node[1]], 
                'green', linestyle='-', linewidth=3, alpha=0.8)
    
    # Annotate nodes
    for i, sensor in enumerate(sensor_positions):
        plt.annotate(f'S{i+1}', (sensor[0], sensor[1]), 
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold', color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    for i, gateway in enumerate(gateways):
        plt.annotate(f'G{i+1}', (gateway[0], gateway[1]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=12, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.8))
    
    plt.annotate('Master', (master_node[0], master_node[1]), 
                xytext=(15, 15), textcoords='offset points', 
                fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    # Styling
    plt.title('🌐 Optimized IoT Network Topology\n(Sensors → Gateways → Master Node)', 
             fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('X Coordinate (meters)', fontsize=14)
    plt.ylabel('Y Coordinate (meters)', fontsize=14)
    plt.xlim(-field_size * 0.05, field_size * 1.05)
    plt.ylim(-field_size * 0.05, field_size * 1.05)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()
    
    return img_base64

def optimize_gateways(field_size: float, n_sensors: int, k_gateways: int, 
                     alpha: float, beta: float) -> Dict[str, Any]:
    """Run the optimization and return results"""
    
    # Generate sensor positions
    sensor_positions = generate_sensors(n_sensors, field_size)
    master_node = np.array([field_size / 2, field_size / 2])
    
    # Define bounds for gateway coordinates
    bounds = [(0, field_size)] * (k_gateways * 2)
    
    # Define fitness wrapper
    def fitness_wrapper(x):
        return compute_fitness(x, sensor_positions, master_node, k_gateways, alpha, beta)
    
    # Run differential evolution optimization
    result = differential_evolution(
        fitness_wrapper,
        bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42
    )
    
    # Extract results
    optimized_gateways = result.x.reshape((k_gateways, 2))
    
    # Calculate load distribution
    load = np.zeros(k_gateways, dtype=int)
    for sensor in sensor_positions:
        dists = np.linalg.norm(optimized_gateways - sensor, axis=1)
        nearest_idx = np.argmin(dists)
        load[nearest_idx] += 1
    
    # Generate visualization
    img_base64 = create_network_visualization(optimized_gateways, sensor_positions, master_node, field_size)
    
    return {
        "fitness": round(result.fun, 2),
        "gateways": optimized_gateways.tolist(),
        "load_distribution": load.tolist(),
        "plot": img_base64,
        "convergence_info": {
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "success": result.success,
            "message": result.message
        },
        "sensor_positions": sensor_positions.tolist(),
        "master_node": master_node.tolist()
    }

# API Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="IoT Node Placement Simulator API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_placement(request: OptimizationRequest):
    """
    Optimize IoT gateway placement using Differential Evolution
    """
    try:
        # Validate inputs
        if request.n_sensors <= 0 or request.k_gateways <= 0 or request.field_size <= 0:
            raise HTTPException(
                status_code=400, 
                detail="All parameters must be positive"
            )
        
        if request.k_gateways > request.n_sensors:
            raise HTTPException(
                status_code=400,
                detail="Number of gateways cannot exceed number of sensors"
            )
        
        if request.alpha < 0 or request.beta < 0:
            raise HTTPException(
                status_code=400,
                detail="Alpha and Beta parameters must be non-negative"
            )
        
        # Clear cache for new optimization
        global _cached_sensors, _cached_field_size
        _cached_sensors = None
        _cached_field_size = None
        
        # Run optimization
        result = optimize_gateways(
            request.field_size,
            request.n_sensors,
            request.k_gateways,
            request.alpha,
            request.beta
        )
        
        return OptimizationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/api/info")
async def get_api_info():
    """Get API information and available endpoints"""
    return {
        "name": "IoT Node Placement Simulator API",
        "version": "2.0.0",
        "framework": "FastAPI",
        "endpoints": {
            "GET /": "Health check",
            "GET /health": "Detailed health check",
            "POST /api/optimize": "Run optimization",
            "GET /api/info": "API information",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "ReDoc API documentation"
        },
        "algorithm": "Differential Evolution",
        "optimization_parameters": {
            "strategy": "best1bin",
            "population_size": 15,
            "max_iterations": 500,
            "mutation_range": [0.5, 1.0],
            "recombination": 0.7
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Use PORT environment variable for deployment compatibility (e.g., Render, Heroku)
    port = int(os.environ.get("PORT", 8000))
    
    # In production, disable reload
    is_development = os.environ.get("ENVIRONMENT", "development") == "development"
    
    print(f"🚀 Starting IoT Node Placement Simulator API on port {port}")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=is_development
    )
