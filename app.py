import spaces
import gradio as gr

def test_api(payload: dict) -> dict:
    return {"received": payload, "message": "Working", "status": "ok"}

@spaces.GPU
def _(): return None

with gr.Blocks() as demo:
    gr.Markdown("Test API")
    gr.api(test_api, api_name="test_api")

demo.queue(default_concurrency_limit=16, max_size=256)
demo.launch(ssr_mode=False)