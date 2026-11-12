import os
import ee
from dotenv import load_dotenv

load_dotenv()

# Set GEE_PROJECT_ID in your .env file (see .env.example).
MY_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "")
if not MY_PROJECT_ID:
    raise EnvironmentError(
        "GEE_PROJECT_ID is not set. Add it to your .env file (see .env.example)."
    )


def initialize_gee():
    """Authenticates and initializes Google Earth Engine.
    Call this function at the start of every script.
    """
    try:
        ee.Initialize(project=MY_PROJECT_ID)
        print("GEE Initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}. Attempting re-authentication...")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)
        print("Authentication complete & GEE Initialized.")

if __name__ == "__main__":
    initialize_gee()