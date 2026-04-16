# Imports
import gradio
import spaces

# Variables

# Functions
def endpoint(payload: dict) -> dict: return {"received": payload, "message": "Working", "status": "ok"}

@spaces.GPU(size="large", duration=1)
def ping():
    print(f"SERVER | Space has been pinged. ☁️")
    return

# Initialize
with gradio.Blocks() as api:
    gradio.api(endpoint, api_name="endpoint")
    gradio.api(ping, api_name="ping")

api.queue(default_concurrency_limit=16, max_size=256).launch(ssr_mode=False)