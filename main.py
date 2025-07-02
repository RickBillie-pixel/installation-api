"""
Installation API - Detects installation symbols from extracted vector data
Identifies electrical outlets, switches, lighting points, etc.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("installation_api")

app = FastAPI(
    title="Installation Symbol Detection API",
    description="Detects installation symbols from extracted vector data",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class InstallationDetectionRequest(BaseModel):
    pages: List[PageData]

@app.post("/detect-installations/")
async def detect_installations(request: InstallationDetectionRequest):
    """
    Detect installation symbols from extracted vector data
    
    Args:
        request: JSON with pages containing drawings and texts
        
    Returns:
        JSON with detected installation symbols for each page
    """
    try:
        logger.info(f"Detecting installation symbols for {len(request.pages)} pages")
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing installation symbols on page {page_data.page_number}")
            
            symbols = _extract_installation_symbols(page_data)
            
            results.append({
                "page_number": page_data.page_number,
                "symbols": symbols
            })
        
        logger.info(f"Successfully detected installation symbols for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting installation symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _extract_installation_symbols(page_data: PageData) -> List[Dict[str, Any]]:
    """
    Extract installation symbols using rule-based approach
    
    Args:
        page_data: Page data containing drawings and texts
        
    Returns:
        List of detected installation symbols with properties
    """
    symbols = []
    
    # Define installation symbol patterns
    installation_patterns = {
        "WCD": ["WCD", "STOPCONTACT", "OUTLET", "SOCKET"],
        "LICHTPUNT": ["LICHTPUNT", "LIGHT", "VERLICHTING", "LAMP"],
        "SCHAKELAAR": ["SCHAKELAAR", "SWITCH", "LIGHT SWITCH"],
        "MV": ["MV", "MECHANISCHE VENTILATIE", "VENTILATIE"],
        "CAI": ["CAI", "TV", "ANTENNE", "CABLE"],
        "THERMOSTAAT": ["THERMOSTAAT", "THERMOSTAT", "TEMP"],
        "ALARM": ["ALARM", "SMOKE", "ROOKMELDER", "BRANDMELDER"],
        "BELL": ["BELL", "DEURBEL", "DOORBELL"]
    }
    
    # Search for installation symbols in text
    for text_item in page_data.texts:
        text_upper = text_item["text"].upper()
        
        for symbol_type, keywords in installation_patterns.items():
            if any(keyword in text_upper for keyword in keywords):
                symbols.append({
                    "type": symbol_type,
                    "position": {
                        "x": (text_item["bbox"]["x0"] + text_item["bbox"]["x1"]) / 2,
                        "y": (text_item["bbox"]["y0"] + text_item["bbox"]["y1"]) / 2
                    },
                    "text": text_item["text"],
                    "bbox": text_item["bbox"],
                    "confidence": 1.0,
                    "reason": f"Text contains {symbol_type} keyword"
                })
    
    # Search for geometric patterns that might represent installation symbols
    for drawing in page_data.drawings:
        for item in drawing["items"]:
            if item["type"] == "rect":
                rect = item["rect"]
                width = rect["width"]
                height = rect["height"]
                
                # Small squares might be electrical outlets
                if 0.005 < width < 0.02 and 0.005 < height < 0.02:
                    symbols.append({
                        "type": "electrical_outlet",
                        "position": {
                            "x": (rect["x0"] + rect["x1"]) / 2,
                            "y": (rect["y0"] + rect["y1"]) / 2
                        },
                        "bbox": rect,
                        "confidence": 0.8,
                        "reason": "Small square pattern"
                    })
            
            elif item["type"] == "curve":
                # Curved lines might represent switches
                symbols.append({
                    "type": "switch",
                    "position": item["p1"],
                    "bbox": {
                        "x0": min(item["p1"]["x"], item["p2"]["x"]),
                        "y0": min(item["p1"]["y"], item["p2"]["y"]),
                        "x1": max(item["p1"]["x"], item["p2"]["x"]),
                        "y1": max(item["p1"]["y"], item["p2"]["y"])
                    },
                    "confidence": 0.6,
                    "reason": "Curved line pattern"
                })
    
    if not symbols:
        logger.warning(f"No installation symbols detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "reason": "No installation symbols detected", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(symbols)} installation symbols on page {page_data.page_number}")
    return symbols

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "installation-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 