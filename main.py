"""
Installation API - Detects installation symbols from extracted vector data
Implements knowledge base rules (Rule 5.6, 8.1-8.4) for installation detection
Identifies electrical outlets, switches, lighting points, and other installation elements
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("installation_api")

# Knowledge Base - Installation Symbol Patterns (Rules 5.6, 8.1-8.3)
INSTALLATION_PATTERNS = {
    "WCD": ["WCD", "STOPCONTACT", "OUTLET", "SOCKET", "WANDCONTACTDOOS"],
    "LICHTPUNT": ["LICHTPUNT", "LIGHT", "VERLICHTING", "LAMP", "PL", "TL", "CEILING LIGHT"],
    "SCHAKELAAR": ["SCHAKELAAR", "SWITCH", "LIGHT SWITCH", "DIMMER", "SW"],
    "MV": ["MV", "MECHANISCHE VENTILATIE", "VENTILATIE", "AFZUIGING", "VENTILATION", "EXTRACTION"],
    "CAI": ["CAI", "TV", "ANTENNE", "CABLE", "ANTENNA", "COAX"],
    "DATA": ["DATA", "UTP", "RJ45", "INTERNET", "NETWERK", "NETWORK"],
    "THERMOSTAAT": ["THERMOSTAAT", "THERMOSTAT", "TEMP", "TEMPERATURE", "TEMP CONTROL"],
    "ALARM": ["ALARM", "SMOKE", "ROOKMELDER", "BRANDMELDER", "SMOKE DETECTOR", "FIRE ALARM"],
    "BELL": ["BELL", "DEURBEL", "DOORBELL", "CHIME", "GONG"],
    "WATERTAP": ["KRAAN", "TAP", "WATER", "WATERTAPPUNT", "WATER SUPPLY"],
    "DRAIN": ["AFVOER", "DRAIN", "WATER DRAIN", "GOOTSTEEN", "SINK"],
    "HRU": ["HRU", "HEAT RECOVERY", "WARMTE TERUGWINNING", "WTW"],
    "HEATING": ["CV", "RADIATOR", "VERWARMING", "HEATING", "BOILER"],
    "PV": ["PV", "SOLAR", "ZONNEPANEEL", "SOLARPANEL"]
}

# Mapping from installation type to standardized codes and labels (Rule 3.1)
INSTALLATION_MAPPING = {
    "WCD": {
        "label_code": "EI01",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_stopcontact",
        "label_en": "Electrical_installation_outlet"
    },
    "LICHTPUNT": {
        "label_code": "EI02",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_lichtpunt",
        "label_en": "Electrical_installation_light"
    },
    "SCHAKELAAR": {
        "label_code": "EI03",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_schakelaar",
        "label_en": "Electrical_installation_switch"
    },
    "MV": {
        "label_code": "HVAC01",
        "label_type": "installation",
        "label_nl": "Klimaatinstallatie_ventilatie",
        "label_en": "HVAC_ventilation"
    },
    "CAI": {
        "label_code": "EI04",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_cai",
        "label_en": "Electrical_installation_cable"
    },
    "DATA": {
        "label_code": "EI05",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_data",
        "label_en": "Electrical_installation_data"
    },
    "THERMOSTAAT": {
        "label_code": "HVAC02",
        "label_type": "installation",
        "label_nl": "Klimaatinstallatie_thermostaat",
        "label_en": "HVAC_thermostat"
    },
    "ALARM": {
        "label_code": "FS01",
        "label_type": "installation",
        "label_nl": "Brandveiligheidstoestel",
        "label_en": "Fire_safety_device"
    },
    "BELL": {
        "label_code": "EI06",
        "label_type": "installation",
        "label_nl": "Elektrainstallatie_bel",
        "label_en": "Electrical_installation_bell"
    },
    "WATERTAP": {
        "label_code": "WI01",
        "label_type": "installation",
        "label_nl": "Waterinstallatie_tap",
        "label_en": "Water_installation_tap"
    },
    "DRAIN": {
        "label_code": "WI02",
        "label_type": "installation",
        "label_nl": "Waterinstallatie_afvoer",
        "label_en": "Water_installation_drain"
    },
    "HRU": {
        "label_code": "HVAC03",
        "label_type": "installation",
        "label_nl": "Klimaatinstallatie_wtw",
        "label_en": "HVAC_heat_recovery"
    },
    "HEATING": {
        "label_code": "HVAC04",
        "label_type": "installation",
        "label_nl": "Klimaatinstallatie_verwarming",
        "label_en": "HVAC_heating"
    },
    "PV": {
        "label_code": "PV01",
        "label_type": "installation",
        "label_nl": "Zonnepanelen_zone",
        "label_en": "PV_zone"
    }
}

app = FastAPI(
    title="Installation Symbol Detection API",
    description="Detects installation symbols from extracted vector data",
    version="1.0.0",
)

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: List[float] = [0, 0, 0]
    bbox: Dict[str, float]

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: List[float] = [0, 0, 0]
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class PageData(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    drawings: Drawings
    texts: List[TextItem]
    is_vector: bool = True
    processing_time_ms: Optional[int] = None

class InstallationDetectionRequest(BaseModel):
    pages: List[PageData]

class InstallationDetectionResponse(BaseModel):
    pages: List[Dict[str, Any]]

# Utility functions
def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_circle(item: dict) -> bool:
    """Check if a curve item forms a circle"""
    if item["type"] != "curve" or not item.get("p1") or not item.get("p2") or not item.get("p3"):
        return False
    
    # For a circle, the three points should be approximately equidistant from center
    center_x = (item["p1"]["x"] + item["p2"]["x"] + item["p3"]["x"]) / 3
    center_y = (item["p1"]["y"] + item["p2"]["y"] + item["p3"]["y"]) / 3
    
    center = {"x": center_x, "y": center_y}
    
    d1 = distance(center, item["p1"])
    d2 = distance(center, item["p2"])
    d3 = distance(center, item["p3"])
    
    # Calculate average distance and check if all points are close to it
    avg_dist = (d1 + d2 + d3) / 3
    tolerance = 0.2  # 20% tolerance
    
    return (abs(d1 - avg_dist) / avg_dist < tolerance and
            abs(d2 - avg_dist) / avg_dist < tolerance and
            abs(d3 - avg_dist) / avg_dist < tolerance)

def find_installation_type_from_text(text: str) -> str:
    """Determine installation type from text using patterns"""
    text_upper = text.upper()
    
    for install_type, patterns in INSTALLATION_PATTERNS.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                return install_type
    
    return None

def get_symbol_shape(item: dict) -> str:
    """Determine symbol shape from drawing item"""
    if item["type"] == "rect":
        rect = item["rect"]
        width = rect["width"]
        height = rect["height"]
        
        # Square-like (aspect ratio close to 1)
        if 0.8 <= width / height <= 1.2:
            return "square"
        else:
            return "rectangle"
    
    elif item["type"] == "curve" and is_circle(item):
        return "circle"
    
    elif item["type"] == "line":
        return "line"
    
    return "unknown"

def is_geometric_pattern_match(shape: str, area: float) -> Dict[str, Any]:
    """Match geometric pattern to installation type"""
    if shape == "circle" and 10 <= area <= 50:
        return {
            "type": "LICHTPUNT",
            "confidence": 0.7,
            "reason": "Circle pattern typical for ceiling light"
        }
    
    elif shape == "square" and 4 <= area <= 25:
        return {
            "type": "WCD",
            "confidence": 0.7,
            "reason": "Small square pattern typical for electrical outlet"
        }
    
    elif shape == "rectangle" and 4 <= area <= 36 and area > 0:
        return {
            "type": "SCHAKELAAR",
            "confidence": 0.6,
            "reason": "Small rectangle pattern typical for switch"
        }
    
    elif shape == "line" and area > 0:
        return {
            "type": "WATERTAP",
            "confidence": 0.5,
            "reason": "Line pattern that may represent water installation"
        }
    
    return None

def calc_item_area(item: dict) -> float:
    """Calculate area of a drawing item"""
    if item["type"] == "rect" and "rect" in item:
        rect = item["rect"]
        return rect["width"] * rect["height"]
    
    elif item["type"] == "curve" and is_circle(item):
        # Estimate circle area
        center_x = (item["p1"]["x"] + item["p2"]["x"] + item["p3"]["x"]) / 3
        center_y = (item["p1"]["y"] + item["p2"]["y"] + item["p3"]["y"]) / 3
        
        center = {"x": center_x, "y": center_y}
        radius = distance(center, item["p1"])
        
        return math.pi * (radius ** 2)
    
    elif item["type"] == "line" and "p1" in item and "p2" in item:
        # Just return line length as pseudo-area
        return distance(item["p1"], item["p2"])
    
    return 0

@app.post("/detect-installations/", response_model=InstallationDetectionResponse)
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
    
    # Step 1: Detect installations from text labels (Rules 5.6, 8.1-8.3)
    logger.info("Detecting installations from text labels...")
    for text_item in page_data.texts:
        text_dict = text_item.dict()
        text_upper = text_dict["text"].upper()
        
        # Search for installation symbols in text
        for symbol_type, keywords in INSTALLATION_PATTERNS.items():
            if any(keyword.upper() in text_upper for keyword in keywords):
                # Create installation symbol
                symbol_info = INSTALLATION_MAPPING.get(symbol_type, {
                    "label_code": "UNKNOWN",
                    "label_type": "installation",
                    "label_nl": "Onbekende_installatie",
                    "label_en": "Unknown_installation"
                })
                
                symbols.append({
                    "type": symbol_type,
                    "label_code": symbol_info["label_code"],
                    "label_type": symbol_info["label_type"],
                    "label_nl": symbol_info["label_nl"],
                    "label_en": symbol_info["label_en"],
                    "position": {
                        "x": (text_dict["bbox"]["x0"] + text_dict["bbox"]["x1"]) / 2,
                        "y": (text_dict["bbox"]["y0"] + text_dict["bbox"]["y1"]) / 2
                    },
                    "text": text_dict["text"],
                    "bbox": text_dict["bbox"],
                    "confidence": 1.0,
                    "reason": f"Text contains {symbol_type} keyword",
                    "source": "text"
                })
                
                # Don't break here - some texts may contain multiple installation references
    
    # Step 2: Detect installations from geometric patterns
    logger.info("Detecting installations from geometric patterns...")
    
    # Process rectangles
    for rect in page_data.drawings.rectangles:
        rect_dict = rect.dict()
        shape = get_symbol_shape(rect_dict)
        area = calc_item_area(rect_dict)
        
        pattern_match = is_geometric_pattern_match(shape, area)
        if pattern_match:
            # Create position from rectangle center
            position = {
                "x": (rect_dict["rect"]["x0"] + rect_dict["rect"]["x1"]) / 2,
                "y": (rect_dict["rect"]["y0"] + rect_dict["rect"]["y1"]) / 2
            }
            
            # Check if we already have a text-based symbol at this position
            duplicate = False
            for existing in symbols:
                if (distance(existing["position"], position) < 20 and 
                    existing["type"] == pattern_match["type"]):
                    duplicate = True
                    break
            
            if not duplicate:
                symbol_info = INSTALLATION_MAPPING.get(pattern_match["type"], {
                    "label_code": "UNKNOWN",
                    "label_type": "installation",
                    "label_nl": "Onbekende_installatie",
                    "label_en": "Unknown_installation"
                })
                
                symbols.append({
                    "type": pattern_match["type"],
                    "label_code": symbol_info["label_code"],
                    "label_type": symbol_info["label_type"],
                    "label_nl": symbol_info["label_nl"],
                    "label_en": symbol_info["label_en"],
                    "position": position,
                    "bbox": rect_dict["rect"],
                    "confidence": pattern_match["confidence"],
                    "reason": pattern_match["reason"],
                    "source": "geometric_pattern",
                    "shape": shape
                })
    
    # Process curves (circles)
    for curve in page_data.drawings.curves:
        curve_dict = curve.dict()
        shape = get_symbol_shape(curve_dict)
        area = calc_item_area(curve_dict)
        
        pattern_match = is_geometric_pattern_match(shape, area)
        if pattern_match:
            # Create position from curve center
            position = {
                "x": (curve_dict["p1"]["x"] + curve_dict["p2"]["x"] + curve_dict["p3"]["x"]) / 3,
                "y": (curve_dict["p1"]["y"] + curve_dict["p2"]["y"] + curve_dict["p3"]["y"]) / 3
            }
            
            # Check if we already have a text-based symbol at this position
            duplicate = False
            for existing in symbols:
                if (distance(existing["position"], position) < 20 and 
                    existing["type"] == pattern_match["type"]):
                    duplicate = True
                    break
            
            if not duplicate:
                symbol_info = INSTALLATION_MAPPING.get(pattern_match["type"], {
                    "label_code": "UNKNOWN",
                    "label_type": "installation",
                    "label_nl": "Onbekende_installatie",
                    "label_en": "Unknown_installation"
                })
                
                symbols.append({
                    "type": pattern_match["type"],
                    "label_code": symbol_info["label_code"],
                    "label_type": symbol_info["label_type"],
                    "label_nl": symbol_info["label_nl"],
                    "label_en": symbol_info["label_en"],
                    "position": position,
                    "bbox": {
                        "x0": min(curve_dict["p1"]["x"], curve_dict["p2"]["x"], curve_dict["p3"]["x"]),
                        "y0": min(curve_dict["p1"]["y"], curve_dict["p2"]["y"], curve_dict["p3"]["y"]),
                        "x1": max(curve_dict["p1"]["x"], curve_dict["p2"]["x"], curve_dict["p3"]["x"]),
                        "y1": max(curve_dict["p1"]["y"], curve_dict["p2"]["y"], curve_dict["p3"]["y"])
                    },
                    "confidence": pattern_match["confidence"],
                    "reason": pattern_match["reason"],
                    "source": "geometric_pattern",
                    "shape": shape
                })
    
    # Process lines for water installations or other linear elements
    for line in page_data.drawings.lines:
        line_dict = line.dict()
        shape = get_symbol_shape(line_dict)
        area = calc_item_area(line_dict)
        
        pattern_match = is_geometric_pattern_match(shape, area)
        if pattern_match and pattern_match["type"] in ["WATERTAP", "DRAIN"]:
            # Create position from line midpoint
            position = {
                "x": (line_dict["p1"]["x"] + line_dict["p2"]["x"]) / 2,
                "y": (line_dict["p1"]["y"] + line_dict["p2"]["y"]) / 2
            }
            
            # Check if we already have a text-based symbol at this position
            duplicate = False
            for existing in symbols:
                if (distance(existing["position"], position) < 20 and 
                    existing["type"] == pattern_match["type"]):
                    duplicate = True
                    break
            
            if not duplicate:
                symbol_info = INSTALLATION_MAPPING.get(pattern_match["type"], {
                    "label_code": "UNKNOWN",
                    "label_type": "installation",
                    "label_nl": "Onbekende_installatie",
                    "label_en": "Unknown_installation"
                })
                
                symbols.append({
                    "type": pattern_match["type"],
                    "label_code": symbol_info["label_code"],
                    "label_type": symbol_info["label_type"],
                    "label_nl": symbol_info["label_nl"],
                    "label_en": symbol_info["label_en"],
                    "position": position,
                    "bbox": {
                        "x0": min(line_dict["p1"]["x"], line_dict["p2"]["x"]),
                        "y0": min(line_dict["p1"]["y"], line_dict["p2"]["y"]),
                        "x1": max(line_dict["p1"]["x"], line_dict["p2"]["x"]),
                        "y1": max(line_dict["p1"]["y"], line_dict["p2"]["y"])
                    },
                    "confidence": pattern_match["confidence"],
                    "reason": pattern_match["reason"],
                    "source": "geometric_pattern",
                    "shape": shape
                })
    
    # Step 3: Associate installation symbols with rooms
    # This would require room data which is not available here
    # In a full implementation, we would link each symbol to the room it's in
    
    if not symbols:
        logger.warning(f"No installation symbols detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "label_code": "UNKNOWN",
            "label_type": "installation",
            "label_nl": "Onbekende_installatie",
            "label_en": "Unknown_installation",
            "reason": "No installation symbols detected", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(symbols)} installation symbols")
    return symbols

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Installation Symbol Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-installations/": "Detect installation symbols",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "installation-api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)