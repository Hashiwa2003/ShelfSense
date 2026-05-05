from openai import OpenAI
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# RAG LOADER
# Loads the ShelfSense knowledge base from the tools folder.
# This document is injected into every tool prompt so the
# agent uses real industry data instead of guessing.
# ============================================================

def load_rag_document(filepath: str = "tools/shelfsense_rag.txt") -> str:
    """
    Loads the ShelfSense RAG knowledge base from disk.
    Returns the full text content, or an empty string if not found.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"  → RAG document loaded: {len(content)} characters from {filepath}")
        return content
    except FileNotFoundError:
        print(f"  ⚠️  RAG document not found at {filepath} — proceeding without it")
        return ""


# Load once at module level so all tools share the same instance
RAG_DOCUMENT = load_rag_document()


# ============================================================
# TOOL 1 — Web Search Tool
# Searches for local events, weather, holidays, and other
# factors that could affect restaurant traffic.
# Called AFTER receiving sales and inventory data.
# RAG used for: multiplier guidance, event impact benchmarks
# ============================================================

def search_local_factors(location: str, date_range_days: int = 7) -> dict:
    """
    Searches for factors that could affect restaurant traffic
    for a given location over the upcoming week.

    Uses the ShelfSense RAG knowledge base to apply accurate
    demand multipliers based on event type and weather conditions.

    Args:
        location: City or neighborhood of the restaurant (e.g. "Austin, Texas")
        date_range_days: Number of days to look ahead (default: 7)

    Returns:
        A dict containing city, date range, events, weather summary,
        holidays, and a recommended traffic multiplier.
    """
    print("ShelfSense is using the search_local_factors tool")

    today = datetime.today()
    end_date = today + timedelta(days=date_range_days)
    date_range_str = f"{today.strftime('%B %d')} to {end_date.strftime('%B %d, %Y')}"

    search_prompt = f"""
    You are a restaurant business analyst with deep knowledge of how local factors
    affect restaurant traffic. Use the reference data below to calibrate your
    multiplier recommendations accurately.

    SHELFSENSE KNOWLEDGE BASE:
    {RAG_DOCUMENT}

    ---

    TASK:
    Analyze the upcoming week for {location} (date range: {date_range_str}).
    Identify:
    1. Major local events (sports games, concerts, festivals, conventions)
    2. Weather forecast (temperature, precipitation, severe weather)
    3. Public holidays or school breaks
    4. Any major construction or road closures
    5. Any other factors that could increase or decrease restaurant traffic

    Use the demand multipliers from Section 3 of the knowledge base to set
    the recommended_multiplier. If multiple factors apply, combine them.

    Return ONLY a JSON object in this exact format:
    {{
        "city": "{location}",
        "date_range": "{date_range_str}",
        "events": [
            {{"name": "event name", "date": "date", "expected_impact": "high/medium/low", "traffic_change_pct": 25}}
        ],
        "weather": {{
            "summary": "brief weather summary",
            "notable_days": [
                {{"date": "date", "condition": "rainy/sunny/etc", "traffic_change_pct": -10}}
            ]
        }},
        "holidays": [
            {{"name": "holiday name", "date": "date", "traffic_change_pct": 20}}
        ],
        "construction_alerts": ["any road closures or construction notes"],
        "overall_traffic_outlook": "busier/slower/normal than average",
        "recommended_multiplier": 1.2,
        "multiplier_reasoning": "explanation of why this multiplier was chosen"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a restaurant business analyst. Use the provided knowledge base to make accurate, data-grounded recommendations. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": search_prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "city": location,
            "date_range": date_range_str,
            "events": [],
            "weather": {"summary": "Unable to retrieve weather data", "notable_days": []},
            "holidays": [],
            "construction_alerts": [],
            "overall_traffic_outlook": "unknown",
            "recommended_multiplier": 1.0,
            "multiplier_reasoning": "Defaulted to 1.0 due to parse error",
            "error": "Could not parse search results"
        }

    print(f"  → Location: {result.get('city')}")
    print(f"  → Date range: {result.get('date_range')}")
    print(f"  → Events found: {len(result.get('events', []))}")
    print(f"  → Traffic outlook: {result.get('overall_traffic_outlook')}")
    print(f"  → Recommended multiplier: {result.get('recommended_multiplier')}")
    print(f"  → Reasoning: {result.get('multiplier_reasoning')}")

    return result


# ============================================================
# TOOL 2 — Demand Forecast Calculator
# Takes past sales data and applies event/weather/holiday
# multipliers to project ingredient demand for the next week.
# Called AFTER the web search tool.
# RAG used for: portion-to-ingredient ratios (Section 1),
#               safety buffer guidelines (Section 2)
# ============================================================

def forecast_demand(
    sales_history: dict,
    local_factors: dict,
    menu_to_ingredients: dict = None
) -> dict:
    """
    Forecasts ingredient demand for the upcoming week based on
    historical sales and local factor multipliers.

    Uses the ShelfSense RAG knowledge base for standard portion
    ratios if no menu_to_ingredients mapping is provided.

    Args:
        sales_history: Dict of menu item -> units sold last week
                       e.g. {"burger": 280, "fries": 190, "milkshake": 95}
        local_factors: Output dict from search_local_factors()
        menu_to_ingredients: Optional dict mapping menu items to ingredient quantities.
                             If not provided, RAG standard ratios will be used.

    Returns:
        A dict with forecasted demand per ingredient for the upcoming week.
    """
    print("ShelfSense is using the forecast_demand tool")

    overall_multiplier = local_factors.get("recommended_multiplier", 1.0)
    events = local_factors.get("events", [])
    weather = local_factors.get("weather", {})
    holidays = local_factors.get("holidays", [])

    mapping_section = ""
    if menu_to_ingredients:
        mapping_section = f"""
    MENU TO INGREDIENT MAPPING (provided by user):
    {json.dumps(menu_to_ingredients, indent=2)}
    """
    else:
        mapping_section = """
    MENU TO INGREDIENT MAPPING:
    No custom mapping provided. Use the standard portion ratios from
    Section 1 of the ShelfSense Knowledge Base below to calculate
    ingredient needs from menu item forecasts.
    """

    forecast_prompt = f"""
    You are a restaurant demand forecasting expert. Use the ShelfSense knowledge
    base below to apply accurate portion ratios, safety buffers, and demand multipliers.

    SHELFSENSE KNOWLEDGE BASE:
    {RAG_DOCUMENT}

    ---

    LAST WEEK'S SALES:
    {json.dumps(sales_history, indent=2)}

    LOCAL FACTORS FOR UPCOMING WEEK:
    - Location: {local_factors.get('city')}
    - Date range: {local_factors.get('date_range')}
    - Events: {json.dumps(events, indent=2)}
    - Weather: {json.dumps(weather, indent=2)}
    - Holidays: {json.dumps(holidays, indent=2)}
    - Overall traffic multiplier: {overall_multiplier}
    - Multiplier reasoning: {local_factors.get('multiplier_reasoning', 'N/A')}

    {mapping_section}

    INSTRUCTIONS:
    1. Apply the traffic multiplier to last week's sales to project menu item demand.
    2. Use the portion ratios (from mapping or Section 1 of knowledge base) to convert
       menu item demand into raw ingredient quantities.
    3. Note which items might see outsized spikes based on the event type
       (e.g. sports games spike burgers and wings more than salads).

    Return ONLY a JSON object in this exact format:
    {{
        "forecast_period": "{local_factors.get('date_range', 'Next 7 days')}",
        "overall_multiplier_applied": {overall_multiplier},
        "menu_item_forecast": {{
            "item_name": {{
                "last_week_units": 0,
                "forecasted_units": 0,
                "change_pct": 0,
                "reason": "explanation referencing specific event or condition"
            }}
        }},
        "ingredient_demand": {{
            "ingredient_name": {{
                "unit": "lbs/units/gallons/etc",
                "forecasted_quantity_needed": 0,
                "based_on_menu_items": ["item1", "item2"],
                "ratio_used": "e.g. 0.33 lbs per burger"
            }}
        }},
        "forecast_notes": "any important caveats, yield adjustments, or assumptions made"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a restaurant demand forecasting expert. Use the provided knowledge base for accurate industry-standard ratios. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": forecast_prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "forecast_period": local_factors.get("date_range", "Next 7 days"),
            "overall_multiplier_applied": overall_multiplier,
            "menu_item_forecast": {},
            "ingredient_demand": {},
            "forecast_notes": "Error parsing forecast — please retry with clearer sales data.",
            "error": "Could not parse forecast results"
        }

    print(f"  → Forecast period: {result.get('forecast_period')}")
    print(f"  → Multiplier applied: {result.get('overall_multiplier_applied')}")
    print(f"  → Menu items forecasted: {len(result.get('menu_item_forecast', {}))}")
    print(f"  → Ingredients forecasted: {len(result.get('ingredient_demand', {}))}")

    return result


# ============================================================
# TOOL 3 — Shopping List Generator
# Cross-references forecasted demand with current inventory
# to produce a prioritized, itemized shopping list.
# Called AFTER the demand forecast tool.
# RAG used for: PAR levels, priority flags, waste benchmarks,
#               shelf life, ordering best practices (Sections 2, 4, 6)
# ============================================================

def generate_shopping_list(
    demand_forecast: dict,
    current_inventory: dict,
    ingredient_prices: dict = None
) -> dict:
    """
    Generates a prioritized shopping list by comparing forecasted
    ingredient demand against current inventory levels.

    Uses the ShelfSense RAG knowledge base for PAR level thresholds,
    waste benchmarks, shelf life data, and ordering best practices.

    Args:
        demand_forecast: Output dict from forecast_demand()
        current_inventory: Dict of ingredient -> current stock
                           e.g. {"ground_beef_lbs": 35, "buns": 40}
        ingredient_prices: Optional dict of ingredient -> price per unit
                           e.g. {"ground_beef_lbs": 4.50, "buns": 0.40}

    Returns:
        A dict with a full itemized shopping list, priorities, and estimated cost.
    """
    print("ShelfSense is using the generate_shopping_list tool")

    prices = ingredient_prices or {}

    shopping_prompt = f"""
    You are a restaurant inventory manager generating a shopping list.
    Use the ShelfSense knowledge base below — specifically Sections 2, 4, and 6 —
    to apply accurate PAR levels, waste thresholds, shelf life limits, and
    ordering best practices.

    SHELFSENSE KNOWLEDGE BASE:
    {RAG_DOCUMENT}

    ---

    FORECASTED INGREDIENT DEMAND (next 7 days):
    {json.dumps(demand_forecast.get('ingredient_demand', {}), indent=2)}

    CURRENT INVENTORY ON HAND:
    {json.dumps(current_inventory, indent=2)}

    KNOWN INGREDIENT PRICES (per unit, if available):
    {json.dumps(prices, indent=2)}

    INSTRUCTIONS:
    For each ingredient:
    1. Calculate order quantity = forecasted need minus current stock
    2. Apply the appropriate safety buffer from Section 2:
       - Perishables (meat, dairy, fresh produce): add 15-20% buffer
       - Dry goods: add 10% buffer
    3. Assign priority using Section 2 thresholds:
       - CRITICAL: stock < 20% of forecasted need
       - HIGH: stock < 50% of forecasted need
       - NORMAL: stock between 50-100% of need
       - SKIP: stock > 150% of need
    4. Flag waste risks using Section 4 shelf life data
    5. Include ordering best practice tips from Section 6 where relevant

    Return ONLY a JSON object in this exact format:
    {{
        "shopping_list": [
            {{
                "ingredient": "name",
                "unit": "lbs/units/etc",
                "current_stock": 0,
                "forecasted_need": 0,
                "order_quantity": 0,
                "order_quantity_with_buffer": 0,
                "buffer_pct_applied": 15,
                "estimated_cost": 0.00,
                "priority": "CRITICAL/HIGH/NORMAL/SKIP",
                "shelf_life_note": "e.g. use within 2 days of delivery",
                "notes": "any flags, warnings, or ordering tips"
            }}
        ],
        "summary": {{
            "total_items_to_order": 0,
            "critical_items": ["list of critical items"],
            "skip_items": ["items with excess stock"],
            "estimated_total_cost": 0.00,
            "cost_available": true
        }},
        "waste_warnings": ["items at risk of spoiling or over-ordering"],
        "stockout_risks": ["items likely to run out before next order"],
        "ordering_tips": ["relevant tips from best practices for this order"]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a restaurant inventory manager. Use the provided knowledge base for accurate PAR levels, shelf life, and ordering best practices. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": shopping_prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "shopping_list": [],
            "summary": {
                "total_items_to_order": 0,
                "critical_items": [],
                "skip_items": [],
                "estimated_total_cost": 0.00,
                "cost_available": False
            },
            "waste_warnings": [],
            "stockout_risks": [],
            "ordering_tips": [],
            "error": "Could not parse shopping list results"
        }

    summary = result.get("summary", {})
    print(f"  → Items to order: {summary.get('total_items_to_order', 0)}")
    print(f"  → Critical items: {summary.get('critical_items', [])}")
    print(f"  → Stockout risks: {result.get('stockout_risks', [])}")
    print(f"  → Estimated total cost: ${summary.get('estimated_total_cost', 0):,.2f}")

    return result


# ============================================================
# EXAMPLE USAGE — chains all three tools with RAG enabled
# ============================================================

if __name__ == "__main__":

    location = "Austin, Texas"

    sales_history = {
        "burger": 280,
        "fries": 190,
        "milkshake": 95,
        "caesar_salad": 60,
        "grilled_chicken_sandwich": 40
    }

    menu_to_ingredients = {
        "burger":                   {"ground_beef_lbs": 0.33, "buns": 1, "lettuce_lbs": 0.05, "tomato_lbs": 0.08},
        "fries":                    {"potatoes_lbs": 0.4},
        "milkshake":                {"milk_gallons": 0.1, "ice_cream_lbs": 0.2},
        "caesar_salad":             {"romaine_lbs": 0.3, "parmesan_lbs": 0.03},
        "grilled_chicken_sandwich": {"chicken_breast_units": 1, "buns": 1, "lettuce_lbs": 0.04}
    }

    current_inventory = {
        "ground_beef_lbs": 35,
        "buns": 40,
        "lettuce_lbs": 4,
        "tomato_lbs": 6,
        "potatoes_lbs": 20,
        "milk_gallons": 4,
        "ice_cream_lbs": 8,
        "romaine_lbs": 2,
        "parmesan_lbs": 1,
        "chicken_breast_units": 15
    }

    ingredient_prices = {
        "ground_beef_lbs": 4.50,
        "buns": 0.40,
        "lettuce_lbs": 1.20,
        "tomato_lbs": 1.50,
        "potatoes_lbs": 0.80,
        "milk_gallons": 3.50,
        "ice_cream_lbs": 5.00,
        "romaine_lbs": 2.00,
        "parmesan_lbs": 8.00,
        "chicken_breast_units": 2.50
    }

    print("\n" + "="*50)
    print("TOOL 1: Searching local factors...")
    print("="*50)
    local_factors = search_local_factors(location)

    print("\n" + "="*50)
    print("TOOL 2: Forecasting demand...")
    print("="*50)
    demand = forecast_demand(sales_history, local_factors, menu_to_ingredients)

    print("\n" + "="*50)
    print("TOOL 3: Generating shopping list...")
    print("="*50)
    shopping_list = generate_shopping_list(demand, current_inventory, ingredient_prices)

    print("\n" + "="*50)
    print("FINAL SHOPPING LIST")
    print("="*50)
    for item in shopping_list.get("shopping_list", []):
        priority = item.get("priority", "NORMAL")
        flag = "🚨" if priority == "CRITICAL" else "⚠️ " if priority == "HIGH" else "✅ " if priority == "SKIP" else "   "
        cost_str = f"  (~${item.get('estimated_cost', 0):.2f})" if item.get('estimated_cost') else ""
        print(f"{flag} {item['ingredient']}: order {item.get('order_quantity_with_buffer', 0)} {item.get('unit', '')}{cost_str}")

    summary = shopping_list.get("summary", {})
    if summary.get("estimated_total_cost"):
        print(f"\nEstimated order total: ${summary['estimated_total_cost']:,.2f}")
    if shopping_list.get("stockout_risks"):
        print(f"\n⚠️  Stockout risks: {', '.join(shopping_list['stockout_risks'])}")
    if shopping_list.get("ordering_tips"):
        print(f"\n💡 Ordering tips:")
        for tip in shopping_list["ordering_tips"]:
            print(f"   • {tip}")
