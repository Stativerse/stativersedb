# Imports
import gradio
import spaces

# Variables

# Functions
def endpoint(payload: dict) -> dict: return {"received": payload, "message": "Working", "status": "ok"}

@spaces.GPU
def _(): return None

# Initialize
with gradio.Blocks() as api:
    gradio.api(endpoint, api_name="endpoint")

api.queue(default_concurrency_limit=16, max_size=256).launch(ssr_mode=False)