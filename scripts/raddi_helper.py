import json
from pathlib import Path

def get_atomic_property(element: str, charge: str, coordination: str, property_name: str):
    """
    Retrieve a specific property (e.g., ionic radius) for an element given its charge and coordination.

    See https://cmd-ml.github.io/

    Args:
        element (str): Element symbol (e.g., 'Ru', 'Cd').
        charge (str): Charge as a string (e.g., '2', '3').
        coordination (str): Coordination number as a string (e.g., 'IV', 'VI', 'VII').
        property_name (str): Property to retrieve (e.g., 'r_ionic', 'r_crystal', 'spin', 'remark').
        json_file (str): Path to the JSON file. Defaults to 'shannon-radii.json'.

    Returns:
        The value of the requested property, or None if not found.
    """
    _data_path = Path(__file__).resolve().parents[1] / "data" / "shannon-radii.json"

    try:
        with open(_data_path, 'r') as f:
            data = json.load(f)

        # Navigate the nested dictionary
        element_data = data.get(element, {})
        charge_data = element_data.get(charge, {})
        coordination_data = charge_data.get(coordination, {})

        return coordination_data.get(property_name)

    except FileNotFoundError:
        print(f"Error: File '{_data_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{_data_path}' is not a valid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


