from dotenv import load_dotenv
load_dotenv()

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Gemini SDK Setup ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    # Handle case where environment might not have google-genai installed for simple local testing
    print("Warning: 'google-genai' not found. Ensure it is installed for production.")
    genai = None
    APIError = type('APIError', (Exception,), {})

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY and genai:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    # This warning is CRITICAL for Cloud Run deployment, where the key MUST be set.
    print("CRITICAL WARNING: GEMINI_API_KEY not found or SDK not available. GenAI step will be skipped/simulated.")
    client = None

# --- Application Setup ---
app = FastAPI(
    title="Kiln Optimization Co-Pilot API",
    version="1.0.0",
    description="API for the 5-step process: Ingestion, Prediction (Sim), Optimization (Sim), and Gen AI Generation.",
)

# --- Pydantic Data Models ---

class SensorDataPoint(BaseModel):
    """Represents a single snapshot of the kiln operational state."""
    timestamp: str
    kiln_inlet_temp: float
    burning_zone_temp: float = Field(..., description="Current burning zone temperature (°C)")
    o2_level: float
    co_level: float
    nox_ppm: float = Field(..., description="Current NOx emissions (ppm)")
    coal_feed_rate: float
    tire_chip_feed: float
    fan_speed_rpm: int
    clinker_free_lime: float = Field(..., description="Current clinker free lime percentage (%)")

class TimeSeriesInput(BaseModel):
    """The main input model, accepting a list of data points for the last hour."""
    data_points: List[SensorDataPoint]

class OptimizationRecommendation(BaseModel):
    """Structure for the final output, combining human and machine-readable data."""
    actionable_recommendation: str = Field(..., description="The clear, concise instruction for the human operator.")
    current_state: SensorDataPoint
    predicted_state: Dict[str, Any]
    optimization_output: Dict[str, Any]

# --- Core Business Logic (Steps 2 & 3 Simulations) ---

def _step_2_simulate_prediction(data: SensorDataPoint) -> Dict[str, Any]:
    """
    Step 2: SIMULATION of the Predictive Forecasting Model (e.g., LSTM).
    Uses the provided stable scenario logic.
    """
    return {
        "forecast_timestamp": "T+30m",
        "predicted_burning_zone_temp": round(data.burning_zone_temp - 3.6, 2), # Slight drop
        "predicted_nox_ppm": round(data.nox_ppm + 5.0, 1), # Slight increase
        "predicted_clinker_free_lime": round(data.clinker_free_lime + 0.05, 2) # Slight increase
    }

def _step_3_simulate_optimization(data: SensorDataPoint, prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: SIMULATION of the Optimization Engine.
    Matches the example goal: Increase TSR (-5% coal, +8% tires, +1.5% fan speed).
    """
    # Apply changes
    optimal_coal_feed_rate = data.coal_feed_rate * (1 - 0.05)
    optimal_tire_chip_feed = data.tire_chip_feed * (1 + 0.08)
    optimal_fan_speed_rpm = int(data.fan_speed_rpm * (1 + 0.015))

    return {
        "optimal_coal_feed_rate": round(optimal_coal_feed_rate, 2),
        "optimal_tire_chip_feed": round(optimal_tire_chip_feed, 2),
        "optimal_fan_speed_rpm": optimal_fan_speed_rpm
    }

def _step_4_generate_recommendation(data: SensorDataPoint, prediction: Dict[str, Any], optimization: Dict[str, Any]) -> str:
    """
    Step 4 & 5: Generation using Gemini 2.5 Flash.
    """
    if not client:
        # Fallback for local testing without key or if key is missing in Cloud Run
        return "GENAI FAILED: API Key not set. Simulated: Prediction shows stable operation. Recommendation: Increase tire chips by 8%, reduce coal by 5%, and raise fan speed 1.5%. This will lower fuel costs while keeping clinker quality high."
        
    current_state_summary = f"Temp: {data.burning_zone_temp}°C, Free Lime: {data.clinker_free_lime}%, NOx: {data.nox_ppm}ppm"
    prediction_summary = f"Predicted Free Lime: {prediction.get('predicted_clinker_free_lime')}%, Predicted NOx: {prediction.get('predicted_nox_ppm')}ppm"

    # DETAILED Prompt for high-quality, justified, and concise output
    prompt = f"""
    You are an AI Co-Pilot for a cement kiln operator. Your task is to translate complex data into a clear, concise, and actionable instruction.
    
    Context:
    - Current Key Status: {current_state_summary}
    - Predicted Future State (in 30 min, if no change): {prediction_summary}
    - Optimal Control Parameters: {json.dumps(optimization)}
    
    Current Feeds: Coal {data.coal_feed_rate}, Tires {data.tire_chip_feed}, Fan {data.fan_speed_rpm}.
    
    Goal: Increase Thermal Substitution Rate (TSR) by using more tire chips, while maintaining clinker quality (free lime < 1.5%) and minimizing NOx.
    
    Task:
    1. Summarize the prediction (e.g., 'stable operation' or 'risk alert').
    2. State the required change in percentage terms for each control.
    3. State the benefit (e.g., lower fuel costs, stabilize temp, reduce NOx).
    
    The final output must be ONLY the recommendation text, without any introductory phrases or markdown formatting.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"temperature": 0.1} # Keeps the generation factual
        )
        # Clean up common GenAI output issues like initial/trailing whitespace
        return response.text.strip().replace('"', '')
    except APIError as e:
        print(f"Gemini API Error: {e}")
        # Ensure the API fails cleanly if the service is down/misconfigured
        raise HTTPException(status_code=500, detail="Error communicating with the Generative AI service.")
    except Exception as e:
        print(f"Unknown error during generation: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during AI generation.")

# --- API Endpoint (Step 1 & 5 Orchestration) ---

@app.post("/optimize_kiln", response_model=OptimizationRecommendation)
async def optimize_kiln(time_series_data: TimeSeriesInput):
    """
    Processes time-series sensor data (last hour) through the 5-step Gen AI pipeline.
    """
    if not time_series_data.data_points:
        raise HTTPException(status_code=400, detail="Input must contain at least one data point.")
        
    # Step 1: Data Ingestion - The Current State is the latest point in the sequence
    current_data = time_series_data.data_points[-1]
    
    # Step 2: Prediction Simulation (uses the current state for simplicity in simulation)
    predicted_state = _step_2_simulate_prediction(current_data)
    
    # Step 3: Optimization Simulation
    optimization_output = _step_3_simulate_optimization(current_data, predicted_state)
    
    # Step 4 & 5: Generation and Output
    actionable_recommendation = _step_4_generate_recommendation(current_data, predicted_state, optimization_output)
    
    return OptimizationRecommendation(
        actionable_recommendation=actionable_recommendation,
        current_state=current_data,
        predicted_state=predicted_state,
        optimization_output=optimization_output
    )