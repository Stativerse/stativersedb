import gradio as gr

def test_api(payload: dict):
    return {
        "received": payload,
        "message": "Working",
        "status": "ok"
    }

with gr.Blocks() as demo:
    gr.Markdown("Test API")
    gr.api(fn = test_api, api_name = "test_api")

demo.queue(default_concurrency_limit = 16)
demo.launch()