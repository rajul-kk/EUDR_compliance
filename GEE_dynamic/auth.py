import ee

# ⚠️ REPLACE THIS with your actual Project ID from the signup page
# It usually looks like "ee-yourname" or "my-project-12345"
MY_PROJECT_ID = "ee-your-project-id"

def initialize_gee():
    """
    Authenticates and initializes Google Earth Engine.
    Call this function at the start of every script.
    """
    try:
        # Try to initialize with the saved token
        ee.Initialize(project=MY_PROJECT_ID)
        print("✅ GEE Initialized successfully!")
    except Exception as e:
        print(f"⚠️ Initialization failed: {e}")
        print("🔄 Attempting to force re-authentication...")
        
        # If it fails, trigger the browser login popup
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)
        print("✅ Authentication complete & GEE Initialized!")

if __name__ == "__main__":
    initialize_gee()