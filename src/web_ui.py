try:
    import gradio as gr
import src.utils as utils
import os
import json
import logging

def _lazy_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        logging.warning("pandas not found, DataFrame editing will not work.")
        return None

def _lazy_import():
    # For lazy loading new modules
    import importlib
    return importlib

# New: Import robot pipeline modules lazily
def get_bom_from_image(image_bytes, prompt_hint=""):
    import src.vision_bom as vision_bom
    return vision_bom.extract_bom(image_bytes, prompt_hint)

def get_skeleton_macro(bom):
    import src.skeleton as skeleton
    return skeleton.generate_skeleton(bom, None)

def get_assembly_macro(bom):
    from src.parts_index import PartIndex
    import src.assembly_builder as assembly_builder
    pi = PartIndex.load()
    if pi.index is None or len(pi.parts) == 0:
        pi.build_index()
    return assembly_builder.build_assembly(bom, pi)

def save_macro_file(macro_code, fname):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", fname)
    with open(path, "w") as f:
        f.write(macro_code)
    return path

def load_bom_to_df(bom):
    pd = _lazy_import_pandas()
    if not pd:
        return None
    # Flatten dict for DataFrame (head, torso, legs, arms)
    flat = []
    for k in ["head", "torso"]:
        val = bom.get(k, {})
        val = val.copy(); val["component"] = k
        flat.append(val)
    for idx, leg in enumerate(bom.get("legs", [])):
        val = leg.copy(); val["component"] = f"leg_{idx}"
        flat.append(val)
    for idx, arm in enumerate(bom.get("arms", [])):
        val = arm.copy(); val["component"] = f"arm_{idx}"
        flat.append(val)
    return pd.DataFrame(flat)

def df_to_bom(df):
    # Rebuild dict from DataFrame
    if df is None:
        return {}
    group = {"head": {}, "torso": {}, "legs": [], "arms": []}
    for _, row in df.iterrows():
        c = row.get("component", "")
        d = row.drop("component").to_dict()
        if c == "head":
            group["head"] = d
        elif c == "torso":
            group["torso"] = d
        elif c.startswith("leg"):
            group["legs"].append(d)
        elif c.startswith("arm"):
            group["arms"].append(d)
    return group

def ui_main():
    with gr.Blocks() as demo:
        gr.Markdown("# Query2CAD Humanoid Robot Pipeline")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="bytes", label="Upload Robot Image")
                prompt_hint = gr.Textbox(label="Prompt hint (optional)", value="")
                extract_btn = gr.Button("Extract BOM")
            with gr.Column():
                bom_df = gr.Dataframe(label="Editable BOM", interactive=True, visible=False)
                skeleton_btn = gr.Button("Generate skeleton", visible=False)
                assembly_btn = gr.Button("Build assembly", visible=False)
                macro_download = gr.File(label="Download Macro", visible=False)
                feedback = gr.Button("👍", visible=True)
        state_bom = gr.State({})
        state_macro = gr.State("")

        def do_extract(image, prompt_hint):
            if not image:
                return gr.update(visible=False), {}, gr.update(visible=False), gr.update(visible=False)
            bom = get_bom_from_image(image, prompt_hint)
            df = load_bom_to_df(bom)
            visible = True if df is not None else False
            return gr.update(visible=visible, value=df), bom, gr.update(visible=visible), gr.update(visible=visible)

        def do_update_df(df):
            # Sync DataFrame to BOM dict
            pd = _lazy_import_pandas()
            if not pd or df is None:
                return {}, gr.update()
            bom = df_to_bom(pd.DataFrame(df))
            return bom, gr.update()

        def do_skeleton(bom):
            macro = get_skeleton_macro(bom)
            path = save_macro_file(macro, "skeleton.FCMacro")
            return gr.update(value=path, visible=True), macro

        def do_assembly(bom):
            macro = get_assembly_macro(bom)
            path = save_macro_file(macro, "assembly.FCMacro")
            return gr.update(value=path, visible=True), macro

        extract_btn.click(do_extract, [image_input, prompt_hint], [bom_df, state_bom, skeleton_btn, assembly_btn])
        bom_df.change(do_update_df, [bom_df], [state_bom, gr.update()])
        skeleton_btn.click(lambda bom: do_skeleton(bom), [state_bom], [macro_download, state_macro])
        assembly_btn.click(lambda bom: do_assembly(bom), [state_bom], [macro_download, state_macro])

        # Feedback button just log for now
        feedback.click(lambda: print("Feedback: thumbs up!"), None, None)

    return demo

if __name__ == "__main__":
    utils.ensure_startup_dirs()
    demo = ui_main()
    demo.launch()
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