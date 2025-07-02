try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

from src.cache import cached_get_answers
from src.prompts import get_parametric_prompt, get_explanation_prompt
from src.logger import get_logger

logger = get_logger("web_ui")

MODEL_OPTIONS = ["gpt-4", "gpt-3.5-turbo"]

def launch_web_ui():
    if not HAS_GRADIO:
        logger.warning("Gradio not installed; web UI unavailable.")
        return

    def infer(query, model, parametric, explain):
        prompt = query
        if parametric:
            prompt = get_parametric_prompt(query)
        macro = cached_get_answers(model, prompt)
        explanation = ""
        if explain:
            explanation = cached_get_answers(model, get_explanation_prompt(macro))
        # TODO: Render CAD image (placeholder)
        return macro, explanation

    with gr.Blocks() as demo:
        gr.Markdown("# Query2CAD Web UI")
        query = gr.Textbox(label="Enter CAD query", lines=3)
        model = gr.Radio(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model")
        parametric = gr.Checkbox(label="Parametric", value=False)
        explain = gr.Checkbox(label="Explain steps", value=False)
        run_btn = gr.Button("Run")
        macro_out = gr.Textbox(label="Generated Macro")
        explanation_out = gr.Textbox(label="Macro Explanation", visible=True)
        with gr.Row():
            thumbs_up = gr.Button("👍 Good")
            thumbs_down = gr.Button("👎 Needs Fix")

        def feedback_good(query, macro):
            from src.retrieval import Retriever
            retriever = Retriever()
            # TODO: embedding placeholder
            retriever.add_example(query, macro, [0]*768)
            return "Feedback recorded!"

        def feedback_bad(query, macro):
            from src.retrieval import Retriever
            retriever = Retriever()
            retriever.add_negative_example(query, "User marked as bad result.")
            return "Feedback recorded!"

        run_btn.click(
            fn=infer,
            inputs=[query, model, parametric, explain],
            outputs=[macro_out, explanation_out]
        )
        thumbs_up.click(fn=feedback_good, inputs=[query, macro_out], outputs=None)
        thumbs_down.click(fn=feedback_bad, inputs=[query, macro_out], outputs=None)

    demo.launch()