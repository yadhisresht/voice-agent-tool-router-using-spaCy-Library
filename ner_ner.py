import requests
import json
import numpy as np
import time
import os
import sys
import re

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("\n[INFO] Real NER (spaCy) model 'en_core_web_sm' loaded successfully.")
except ImportError:
    SPACY_AVAILABLE = False
    print("\n[WARNING] spaCy not installed. Falling back to simple keyword matching.")
except OSError:
    SPACY_AVAILABLE = False
    print("\n[WARNING] spaCy model not found. Falling back to non-NLP Regex/Keyword methods.")

try:
    import xgboost as xgb 
    XGB_AVAILABLE = True
    print("[INFO] XGBoost imported successfully.")
except ImportError:
    XGB_AVAILABLE = False
    print("[INFO] XGBoost not found.")

API_KEY = "INSERT YOUR API KEY" 
EMBEDDING_MODEL = "text-embedding-004"
BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
EMBEDDING_API_URL = f"{BASE_API_URL}{EMBEDDING_MODEL}:embedContent?key={API_KEY}"
DB_FILE = "tools_db_simulated_final.json"
XGB_MODEL_FILE = "xgboost_model.json"
TOP_N_RESULTS = 5 

TOOL_FUNCTION_SPECS = {
    "Tool: Local Weather Forecast API": {
        "description": "Retrieves real-time weather information for a specified geographical location.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "location": {"type": "STRING", "description": "The city, region, or address to retrieve the weather for. Example: Bengaluru, New York City, or zip code 90210."}
            },
            "required": ["location"]
        }
    },
    "Tool: GPS Navigation Router": {
        "description": "Calculates a driving route.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "destination": {"type": "STRING", "description": "The final destination address or POI name."},
                "start_location": {"type": "STRING", "description": "The starting point, if specified. Default to current vehicle location if null."}
            },
            "required": ["destination"]
        }
    },
    "Tool: Hands-Free Calling Dialer": {
        "description": "Initiates a phone call to a contact.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "contact_name": {"type": "STRING", "description": "The name of the contact or phone number to call."},
            },
            "required": ["contact_name"]
        }
    },
    "Tool: Infotainment Music Streamer": {
        "description": "Controls music playback or selects content.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "enum": ["play", "pause", "skip"], "description": "The command for playback control."},
                "item_name": {"type": "STRING", "description": "The name of the song, artist, or playlist to search for and play. Can be null for simple controls like 'pause'."}
            },
            "required": ["action"]
        }
    }
}

INITIAL_TOOLS = {
    "Tool: GPS Navigation Router": "Specialized service that calculates the fastest or shortest driving route to a user-specified destination. It uses real-time road data to provide turn-by-turn spoken directions, dynamic re-routing around closures, and an accurate Estimated Time of Arrival (ETA).",
    "Tool: Real-Time Traffic Reporter": "Monitors and analyzes live data streams from various sources (sensors, user reports) to detect current traffic congestion, accidents, and road closures. It alerts the driver to delays and suggests faster alternative routes on the fly.",
    "Tool: Local Weather Forecast API": "Integrates with meteorological services to retrieve current conditions, hourly forecasts, and precipitation warnings for the vehicle's current location, destination, or any requested city. Essential for trip safety and planning.",
    "Tool: User Profile Lookup Service": "Accesses and manages secure, personalized user data stored locally or in the cloud. This includes retrieving saved home/work addresses, vehicle preferences, favorite contacts, and personalized radio presets.",
    "Tool: Vehicle Diagnostic System (DTC)": "Directly interfaces with the car's On-Board Diagnostics (OBD-II) port to read and interpret Diagnostic Trouble Codes (DTCs). It reports critical engine health issues, fluid levels, and system malfunctions to the driver in clear, understandable language.",
    "Tool: Infotainment Music Streamer": "Manages playback controls (play, pause, skip), allows selection of integrated music streaming services (Spotify, Pandora), and facilitates browsing of local media or connected devices, all via voice command.",
    "Tool: Hands-Free Calling Dialer": "Utilizes Bluetooth connectivity to synchronize with a user’s mobile phone, enabling the driver to initiate, accept, or reject phone calls, view recent call history, and dial contacts using voice commands only, ensuring driver focus remains on the road.",
    "Tool: Climate Control Regulator": "Allows precise voice control over the car's Heating, Ventilation, and Air Conditioning (HVAC) system. Functions include setting specific cabin temperatures, adjusting fan speed, controlling air direction, and activating defrosters.",
    "Tool: Nearby POI (Point of Interest) Finder": "A specialized geospatial search tool that locates relevant points of interest along a route or near the vehicle's current position, such as fuel stations, charging points, restaurants, rest areas, and parking facilities.",
    "Tool: Voice Command Parser": "The core NLP module responsible for transcribing spoken audio into text, interpreting the user's intent, and translating that intent into structured system commands for the appropriate vehicle function or application.",
    "Tool: Reminder and Calendar Scheduler": "Creates and manages spoken reminders and integrates with the user's external calendar service (e.g., Google Calendar). It can proactively alert the driver to upcoming events, meetings, or time-based tasks.",
    "Tool: Security System Lock/Unlock": "A low-level vehicle interface tool that executes secure commands to remotely lock or unlock all car doors and the trunk. It often requires secondary voice biometric authentication for security.",
    "Tool: Fuel Consumption Tracker": "Calculates and displays real-time and historical data regarding the vehicle's fuel economy (MPG or L/100km). It helps drivers monitor driving habits and efficiency trends over time.",
    "Tool: Tire Pressure Monitoring System": "Continuously monitors the pressure in all four tires (and spare, if applicable) using built-in sensors. It alerts the driver immediately if pressure drops below a safe threshold, mitigating accident risk.",
    "Tool: Voice Biometric Authenticator": "A high-security tool that verifies the driver's identity based on their unique voice signature. Used for accessing sensitive functions like remote start, profile changes, or financial transactions.",
    "Tool: Highway Toll Calculator": "Provides pre-trip and dynamic calculation of estimated toll road fees for an entire planned route. Essential for long-distance travel budgeting and navigation planning.",
    "Tool: Parking Assist Activator": "Initiates the automated self-parking sequence for parallel or perpendicular parking spaces. The system takes control of the steering and often the speed while the driver monitors the process.",
    "Tool: Geofence Alarm System": "Allows the user to define a custom geographic boundary. It triggers an alert notification to the user's mobile device if the vehicle enters or exits the defined area, commonly used for tracking teens or recovering stolen vehicles.",
    "Tool: Software Update Manager": "Manages the Over-The-Air (OTA) software delivery process. It checks for new firmware versions, downloads updates securely in the background, and schedules the installation for a safe, appropriate time.",
    "Tool: E-Call Emergency Responder": "A critical safety system that automatically places an emergency call to public safety answering points (PSAPs) after a severe accident is detected, transmitting the vehicle's precise GPS location and severity data.",
    "Tool: Maintenance Schedule Notifier": "Tracks accumulated mileage, time since last service, and component life cycles. It proactively notifies the driver when scheduled maintenance (e.g., oil change, tire rotation) is due based on manufacturer recommendations.",
    "Tool: Internet Browser Access": "Enables limited and safe access to web browsing on the vehicle's infotainment screen while the vehicle is stationary or via voice commands. Safety protocols restrict its use while driving.",
    "Tool: Live News Feed Aggregator": "Pulls dynamic content from trusted news sources, filtering headlines and summarizing articles based on user preferences (e.g., finance, sports, local). It can read summaries aloud while driving.",
    "Tool: Currency Converter": "Performs real-time foreign currency conversions based on the latest exchange rates. Highly useful for international travelers to calculate prices and budgets instantly.",
    "Tool: Roadside Assistance Requester": "In the event of a breakdown, this tool sends the vehicle's current location, VIN, and immediate diagnostic data directly to a contracted roadside assistance provider for rapid support dispatch.",
    "Tool: Voice Assistant Settings Manager": "Provides an interface for adjusting the operational parameters of the voice assistant itself, including wake word sensitivity, speech speed, voice gender selection, and accent recognition.",
    "Tool: User Destination History Log": "Maintains a chronological record of all recently visited and frequently accessed navigation destinations. This speeds up future routing by suggesting common places quickly.",
    "Tool: Traffic Incident Reporter (Waze-like)": "Allows the driver to verbally report road hazards, police presence, or accidents to a shared community mapping service, contributing to real-time traffic data accuracy for other drivers.",
    "Tool: Vehicle Position GPS Tracker": "Provides the vehicle's exact current latitude, longitude, and altitude coordinates. It is often used for location sharing or integrating the car's position into external applications.",
    "Tool: Sunroof and Window Controller": "Manages the powered opening and closing of the sunroof and all four electric windows through simple, segmented voice commands (e.g., 'open window halfway', 'close sunroof completely').",
    "Tool: Language Translation Service": "Provides real-time, bidirectional translation of spoken words between two selected languages. Highly valuable in international border crossings or when interacting with non-native speakers.",
    "Tool: Electric Vehicle Charging Station Finder": "A specialized POI finder that locates and filters electric vehicle charging stations based on connector type (CCS, Tesla, CHAdeMO) and real-time availability status.",
    "Tool: Battery Health Monitor (EV/Hybrid)": "Continuously tracks and reports the state of health (SOH) and degradation level of the high-voltage traction battery pack in electric and hybrid vehicles, alongside current charge level and estimated range.",
    "Tool: Headlight Control System": "Manages the vehicle's lighting systems, including automatic switching between low beams and high beams, activating fog lights, and adjusting the intensity of ambient cabin lighting.",
    "Tool: Wi-Fi Hotspot Manager": "Activates and configures the in-car Wi-Fi network, allowing passengers to connect their devices to the internet using the vehicle's cellular modem connection.",
    "Tool: Vehicle Manual Search Engine": "A localized search utility that indexes and queries the car's complete digital owner's manual. It provides immediate, context-specific instructions related to vehicle features or warnings.",
    "Tool: Remote Engine Starter": "A security-gated function that allows the driver to remotely start or stop the vehicle's engine or climate control system via a voice command, typically requiring high-security biometric verification.",
    "Tool: Local Time and Date Reader": "Provides the precise current time and calendar date, synchronized to GPS data and adjusted automatically for time zones as the vehicle travels.",
    "Tool: Public Transport Integration": "Provides real-time schedules, route maps, and service alerts for local public transit options (bus, train, subway) that may be near the driver's final destination.",
    "Tool: Fuel Price Comparator": "Queries nearby gas stations and displays a comparative list of current fuel prices (e.g., Regular, Premium, Diesel) to help the driver find the most economical option.",
    "Tool: Voice Memo Recorder": "Records and securely saves short audio notes or spoken reminders directly to the vehicle's local storage or a synchronized cloud account for later playback.",
    "Tool: Seat Position Memory Controller": "Stores and recalls specific seating, mirror, and steering wheel positions for multiple drivers based on profile authentication or voice command.",
    "Tool: Child Lock System Activator": "A safety tool that enables or disables the rear power window and door safety locks, preventing child passengers from opening doors or windows while the vehicle is in motion.",
    "Tool: Garage Door Opener Integration": "Interfaces with compatible smart home systems (e.g., HomeLink) to open and close garage doors and gates using a voice command or proximity-based trigger.",
    "Tool: Nearby Events and Attractions Calendar": "Searches for and presents upcoming local events, festivals, concerts, and attractions near the vehicle's location or a planned destination, providing details and ticket information.",
    "Tool: Vehicle Status Checker (Doors, Lights)": "Provides an immediate status report of the vehicle's physical state, confirming if all doors are securely closed, headlights are off, and windows are rolled up.",
    "Tool: Speed Limit Warning System": "Uses mapping data and camera recognition to identify the current posted speed limit and provides visual and auditory warnings if the vehicle exceeds the limit by a configurable threshold.",
    "Tool: Valet Mode Activator": "Engages a restricted driving profile that limits vehicle top speed, disables access to personal data, and locks the trunk, useful when handing the car over to a third party.",
    "Tool: Favorite Contact Messaging Sender": "Sends simple, pre-defined text messages (e.g., 'Running late', 'Arrived safe') to a list of pre-selected favorite contacts using only voice commands.",
    "Tool: Emergency Contact List Access": "Displays and initiates calls to a list of designated emergency contacts, such as family members or medical providers, accessible quickly in non-E-Call emergency situations."
}

def load_database():
    if not os.path.exists(DB_FILE):
        return None, None 
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)    
        loaded_tools = data.get('papers', {})
        loaded_embeddings_raw = data.get('embeddings', {})
        loaded_embeddings = {}
        for title, vec_list in loaded_embeddings_raw.items():
            loaded_embeddings[title] = np.array(vec_list)
        print(f"Loaded {len(loaded_tools)} tools from {DB_FILE}.")
        return loaded_tools, loaded_embeddings
    except Exception as e:
        print(f"Error loading database from {DB_FILE}: {e}")
        return None, None

def save_database(tools, embeddings):
    serializable_embeddings = {
        title: vec.tolist() for title, vec in embeddings.items()
    }    
    data = {
        'papers': tools,
        'embeddings': serializable_embeddings
    }    
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Database successfully saved to {DB_FILE}.")
    except Exception as e:
        print(f"Error saving database to {DB_FILE}: {e}")

def get_embedding(text):
    if API_KEY == "AIzaSyASRU-gsPRUnc5yPSYzk6P4DoA_4KJH-1U":
        return np.random.rand(768) 
    
    payload = {"content": {"parts": [{"text": text}]}}
    delay = 1
    
    for _ in range(3): 
        try:
            response = requests.post(EMBEDDING_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status() 
            values = response.json().get('embedding', {}).get('values')
            if values:
                return np.array(values)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                time.sleep(delay)
                delay *= 2
            else:
                print(f"HTTP Error: {e.response.status_code}")
                return None
        except Exception as e:
            print(f"Request Error: {e}")
            return None
    return None

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norm if norm else 0

def extract_features(query, tool_name, raw_similarity_score):
    F1 = raw_similarity_score
    query_lower = query.lower()
    is_location = any(kw in query_lower for kw in ["route", "map", "destination", "navigate", "traffic"])
    tool_description = INITIAL_TOOLS.get(tool_name, "").lower()
    is_diagnostic_tool = 1.0 if "diagnostic trouble codes" in tool_description or "engine health" in tool_description else 0.0
    is_music = 1.0 if any(kw in query_lower for kw in ["music", "song", "play", "skip"]) else 0.0
    return np.array([F1, 1.0 if is_location else 0.0, is_diagnostic_tool, 1.0 if is_music else 0.0, 0.0])

def mock_ltr_ranker(feature_vector):
    F1 = feature_vector[0]
    raw_logit = (F1 * 1.5) - 0.6 
    confidence_score = 1 / (1 + np.exp(-35 * raw_logit)) 
    return confidence_score

def load_xgboost_model():
    if not XGB_AVAILABLE: return None
    try:
        model = xgb.XGBRanker()
        model.load_model(XGB_MODEL_FILE)
        print("[INFO] Successfully loaded XGBoost LTR Model.")
        return model
    except Exception:
        return None

def train_mock_xgboost_model():
    return lambda features: mock_ltr_ranker(features)

ENTITY_MAPPING = {
    # General Location entities
    "GPE": "LOCATION",  
    "LOC": "LOCATION",  
    "FAC": "LOCATION",  
    "PERSON": "CONTACT_NAME",
    "ORG": "BUSINESS_POI", 
    "DATE": "DATE",
    "TIME": "TIME_SLOT"
}

def clean_entity(entity_text):
    if not entity_text:
        return None
    # Enhanced cleanup pattern to remove common prepositions and introductory words
    CLEANUP_PATTERN = r'^\s*(?:to|from|for|at|in|the|a|an|and|or|go|navigate|route|what is the weather)\s+'
    cleaned = re.sub(CLEANUP_PATTERN, '', entity_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'[?.,!]\s*$', '', cleaned).strip()
    
    return cleaned if cleaned else None

def extract_entities_with_spacy(query):

    recognized_entities = []

    # 1. Standard spaCy NER Extraction
    if SPACY_AVAILABLE:
        doc = nlp(query)
        
        for ent in doc.ents:
            spacy_type = ent.label_
            app_type = ENTITY_MAPPING.get(spacy_type)
            
            cleaned_value = clean_entity(ent.text)
            
            if not cleaned_value:
                continue

            if app_type:
                final_type = app_type
                # Heuristics: If a general location is preceded by route/go to, treat it as a DESTINATION
                if app_type in ["LOCATION", "BUSINESS_POI"] and re.search(r'\b(to|go to|take me)\b', ent.text, re.IGNORECASE):
                    final_type = "DESTINATION"
                
                recognized_entities.append({
                    "entity_type": final_type,
                    "value": cleaned_value,
                    "spacy_label": spacy_type
                })
    
    # 2. Enhanced Regex Fallbacks for Navigation (Fixes Issue 2 - Non-standard route phrasing)
    
    # Regex A: 'from X to Y' (Standard)
    nav_match_a = re.search(r'\bfrom\s+(.+?)\s+to\s+(.+)\b', query, re.IGNORECASE)
    # Regex B: 'to X from Y' (Common variation)
    nav_match_b = re.search(r'\bto\s+(.+?)\s+from\s+(.+)\b', query, re.IGNORECASE)
    
    if nav_match_a:
        start_loc = clean_entity(nav_match_a.group(1).strip())
        dest_loc = clean_entity(nav_match_a.group(2).strip())
        if start_loc:
             recognized_entities.append({"entity_type": "LOCATION", "value": start_loc, "spacy_label": "REGEX_START"})
        if dest_loc:
             recognized_entities.append({"entity_type": "DESTINATION", "value": dest_loc, "spacy_label": "REGEX_DESTINATION"})
            
    elif nav_match_b: 
        dest_loc = clean_entity(nav_match_b.group(1).strip())
        start_loc = clean_entity(nav_match_b.group(2).strip())
        if start_loc:
             recognized_entities.append({"entity_type": "LOCATION", "value": start_loc, "spacy_label": "REGEX_START"})
        if dest_loc:
             recognized_entities.append({"entity_type": "DESTINATION", "value": dest_loc, "spacy_label": "REGEX_DESTINATION"})

    # Regex C: Simple Navigation Pattern - Handles 'route to X' or ambiguous 'to X to Y' as one destination
    if not any(e['entity_type'] in ['DESTINATION'] for e in recognized_entities):
        simple_dest_match = re.search(r'\b(?:route|navigate|go to|take me to)\s+(?:to\s+)?(.+)\b', query, re.IGNORECASE)
        if simple_dest_match:
            dest_loc = clean_entity(simple_dest_match.group(1).strip())
            if dest_loc:
                # Cleanup to remove trailing question marks or junk
                dest_loc = re.sub(r'\s+(?:is|what|how|what will|will be).*$', '', dest_loc).strip()
                if dest_loc:
                    recognized_entities.append({"entity_type": "DESTINATION", "value": dest_loc, "spacy_label": "REGEX_SIMPLE_DEST"})

    # 3. Dedicated Weather Location Regex (Fixes Issue 3 - Weather location priority)
    weather_loc_match = re.search(r'\b(?:weather|forecast)\s+in\s+(.+)\b', query, re.IGNORECASE)
    if weather_loc_match:
        loc = clean_entity(weather_loc_match.group(1).strip())
        # Cleanup to remove trailing question words or entities
        loc = re.sub(r'\s+(?:is|what|how|what will|will be).*$', '', loc).strip()
        if loc:
            # Use a unique entity type for weather to prioritize it later
            recognized_entities.append({"entity_type": "WEATHER_LOCATION", "value": loc, "spacy_label": "REGEX_WEATHER_LOC"})


    # Filter out empty or junk entities
    return [ent for ent in recognized_entities if ent['value'] is not None and len(ent['value'].split()) > 0]


def ner_based_plan_tasks(query):
    entities = extract_entities_with_spacy(query)
    
    start_location = None
    destination = None
    contact_name = None
    
    # Prioritize entities based on type hierarchy (Destinations are often ambiguous LOCs)
    for ent in entities:
        if ent['entity_type'] == 'DESTINATION' and destination is None:
            destination = ent['value']
        elif ent['entity_type'] == 'LOCATION' and start_location is None:
            start_location = ent['value']
        elif ent['entity_type'] == 'CONTACT_NAME' and contact_name is None:
            contact_name = ent['value']
        # If we still lack a destination, check for a general location (could be the destination)
        elif ent['entity_type'] in ['LOCATION', 'BUSINESS_POI'] and destination is None:
            destination = ent['value']
            
    plan = []
    
    tool_map = {
        "weather": "Tool: Local Weather Forecast API",
        "navigate": "Tool: GPS Navigation Router",
        "call": "Tool: Hands-Free Calling Dialer",
        "music": "Tool: Infotainment Music Streamer"
    }
    
    last_destination = "New York" # Placeholder for state persistence logic
    
    query_lower = query.lower()

    if re.search(r'(navigate|route|go to|take me to|distance|from\s+.+\s+to|to\s+.+\s+from)', query_lower):
        
        if destination:
            # Handles all navigation intents
            plan.append({
                "tool_name": tool_map["navigate"],
                "invocation_order": 1,
                "parameters": {
                    "start_location": start_location or "current vehicle location",
                    "destination": destination
                }
            })
            
    elif re.search(r'(call|phone|ring)', query_lower):        
        if contact_name:
            plan.append({
                "tool_name": tool_map["call"],
                "invocation_order": 1,
                "parameters": {
                    "contact_name": contact_name
                }
            })
            
    elif re.search(r'(weather|forecast)', query_lower):
        
        # FIX FOR ISSUE 3: Prioritize dedicated weather location entity
        weather_loc_entity = next((ent for ent in entities if ent['entity_type'] == 'WEATHER_LOCATION'), None)

        if weather_loc_entity:
            location = weather_loc_entity['value']
        else:
            # Fallback to route locations
            location = destination or start_location 
        
        if not location and re.search(r'\b(?:there|that location|the destination)\b', query_lower):
            # If user asks for weather 'there' but no location is found, use a known state.
            location = last_destination 
        
        if location:
            plan.append({
                "tool_name": tool_map["weather"],
                "invocation_order": 1,
                "parameters": {
                    "location": location
                }
            })
        else:
            # Final fallback to vehicle location
            plan.append({
                "tool_name": tool_map["weather"],
                "invocation_order": 1,
                "parameters": {
                    "location": "current vehicle location" 
                }
            })

    elif re.search(r'(play|pause|skip|music|song|artist)', query_lower):
        action = "play"
        if 'pause' in query_lower: action = "pause"
        elif 'skip' in query_lower: action = "skip"

        item_name = None
        
        if action == "play":
            match = re.search(r'\bplay\s+(.+)$', query_lower)
            if match:
                item_name = clean_entity(match.group(1))

        plan.append({
            "tool_name": tool_map["music"],
            "invocation_order": 1,
            "parameters": {
                "action": action,
                "item_name": item_name or "last played station/song"
            }
        })
    
    if not plan:
        return [{"error": "Real NER analysis found no matching intent or extractable entities for the query."}]
    
    return plan

def resolve_tools_for_query(query, embeddings, ltr_model_predictor):

    
    print("\n--- STAGE 1: Tool Selection & Ranking (LTR) ---")
    
    query_vector = get_embedding(query)
    if query_vector is None: 
        print("Tool Resolution failed due to vector generation error.")
        return

    results = []
    feature_matrix = []
    for tool_name, vector in embeddings.items():
        raw_similarity = cosine_similarity(query_vector, vector)
        feature_vector = extract_features(query, tool_name, raw_similarity)
        results.append({'tool_name': tool_name, 'raw_similarity': raw_similarity})
        feature_matrix.append(feature_vector)
    feature_matrix = np.array(feature_matrix)

    ranking_source = "DISCRIMINATIVE SIMULATED LTR Model."
    if XGB_AVAILABLE and callable(getattr(ltr_model_predictor, 'predict', None)):
        try:
            final_scores = ltr_model_predictor.predict(feature_matrix)
            ranking_source = "TRAINED XGBOOST LTR Model."
        except:
            final_scores = [mock_ltr_ranker(features) for features in feature_matrix]
    else:
        final_scores = [mock_ltr_ranker(features) for features in feature_matrix]

    final_results = sorted([(score, results[i]['tool_name']) for i, score in enumerate(final_scores)], reverse=True)
    
    for i, (score, tool_name) in enumerate(final_results[:TOP_N_RESULTS]):
        print(f"Rank {i + 1} (Score: {score:.4f}): {tool_name}")
    print(f"(Ranking used: {ranking_source})")
    print("-" * 70)


    print("\n--- STAGE 2: REAL NER Agent Planning (Entity-Based Execution) ---")
    
    execution_plan = ner_based_plan_tasks(query)

    print("\n**[DEBUG] All Entities Found by NER/Fallback:**")
    final_spacy_entities = extract_entities_with_spacy(query)
    if final_spacy_entities:
        for ent in final_spacy_entities:
            print(f"  - App Type: {ent['entity_type']} -> Value: '{ent['value']}' (Source: {ent['spacy_label']})")
    else:
          print("  - No entities found by Standard NER, Token Lookahead, or Regex Fallback.")

    if isinstance(execution_plan, list) and not execution_plan[0].get("error"):
        
        print("\nGENERATED EXECUTION PLAN (Real NER)")
        
        print(f"\nPlan contains {len(execution_plan)} sequential tasks:")
        

        for task in execution_plan:
            tool_name = task.get('tool_name', 'N/A')
            order = task.get('invocation_order', 'N/A')
            params = task.get('parameters', {})
            
            print(f"\nTask {order}: {tool_name.split(': ')[1]}")
            print(f"  > Full Tool Name: {tool_name}")
            print(f"  > Arguments (Resolved): {json.dumps(params, indent=4)}")

    else:
        print("\n**FAILURE: COULD NOT GENERATE EXECUTION PLAN**")
        print(f"Error details: {execution_plan[0].get('error', 'Unknown Error')}")
        
    print("-" * 70, "\n")


def add_new_tool_to_db(tool_database, precomputed_embeddings):
    """Adds a new tool and its embedding to the live database and saves it permanently."""
    print("\n--- ADD NEW TOOL ---")
    tool_name = input("Tool Name (e.g., 'Tool: New Diagnostics'): ").strip()
    description = input("Tool Description (What it does): ").strip()

    if not tool_name or not description:
        print("Aborted. Name and Description are required.")
        return

    embedding_text = f"{tool_name}: {description}"
    vector = get_embedding(embedding_text)
    
    if vector is not None:
        tool_database[tool_name] = description
        precomputed_embeddings[tool_name] = vector
        
        save_database(tool_database, precomputed_embeddings) 
        print(f"Added '{tool_name}'. Total tools: {len(tool_database)}\n")
    else:
        print("Failed to add tool. Check your API key and permissions.\n")


TOOLS, embeddings = load_database()
initial_load_needed = False
LTR_PREDICTOR = None

if API_KEY == "AIzaSyCABJrbuza3AKuFhbp0gZKKi-jdGaL7eck":
    print("\n[WARNING] API Key is the default placeholder. Tool embedding calls will be skipped/mocked.")
    API_KEY = "AIzaSyASRU-gsPRUnc5yPSYzk6P4DoA_4KJH-1U" 

if TOOLS is None or embeddings is None:
    TOOLS = INITIAL_TOOLS 
    embeddings = {}
    initial_load_needed = True

if API_KEY != "AIzaSyASRU-gsPRUnc5yPSYzk6P4DoA_4KJH-1U":
    if initial_load_needed or len(TOOLS) > len(embeddings):
        print("Pre-computing embeddings for new/missing tools...")
        temp_embeddings = {}
        for tool_name, description in TOOLS.items():
            if tool_name not in embeddings: 
                vector = get_embedding(f"{tool_name}: {description}")
                if vector is not None: temp_embeddings[tool_name] = vector
                else: print(f"Skipping tool: {tool_name} due to embedding failure.")

        embeddings.update(temp_embeddings)
        if not embeddings: sys.exit(1)
        if temp_embeddings or initial_load_needed: save_database(TOOLS, embeddings)
    
    if XGB_AVAILABLE: LTR_PREDICTOR = load_xgboost_model()
    if LTR_PREDICTOR is None: LTR_PREDICTOR = train_mock_xgboost_model()
            
    print(f"Setup complete. Tool database size: {len(TOOLS)} tools.")
    print("\n--- INTERACTIVE TOOL RESOLUTION (NER-Based Agent System) ---")
    print("Stage 2 is now using REAL NER ")
    print("Commands: Type a query / 'add' / 'quit'")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            save_database(TOOLS, embeddings)
            break

        if user_input.lower() == 'add':
            add_new_tool_to_db(TOOLS, embeddings)
        elif user_input:
            resolve_tools_for_query(user_input, embeddings, LTR_PREDICTOR)

    print("\nTool Resolution session complete. Goodbye !")
else:
    print("\nWARNING: API Key is missing. Stage 1 (LTR) will use random scores, and new tool additions will use random embeddings.")
    print("Please set a valid API key to enable real vector embedding functionality.")
    
    if initial_load_needed or len(TOOLS) > len(embeddings):
        for tool_name, description in TOOLS.items():
            if tool_name not in embeddings:
                embeddings[tool_name] = np.random.rand(768)
        if initial_load_needed: save_database(TOOLS, embeddings)

