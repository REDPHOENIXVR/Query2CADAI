try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None  # type: ignore
    HAS_GRADIO = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    openai = None
    HAS_OPENAI = False

import src.utils as utils
import os
import json
import time
import logging
import inspect
from datetime import datetime
from src.parts_index import PartIndex
import src.assembly_builder as assembly_builder

def _lazy_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        logging.warning("pandas not found, DataFrame editing will not work.")
        return None

def _lazy_import():
    import importlib
    return importlib

# Set up global PartIndex cache
pi_global = PartIndex.load()
if pi_global.index is None or len(pi_global.parts) == 0:
    pi_global.build_index()

# New: Import robot pipeline modules lazily
def get_bom_from_image(image_bytes, prompt_hint=""):
    import src.vision_bom as vision_bom
    return vision_bom.extract_bom(image_bytes, prompt_hint)

def get_skeleton_macro(bom, param=None):
    import src.skeleton as skeleton
    return skeleton.generate_skeleton(bom, param)

def get_assembly_macro(bom):
    return assembly_builder.build_assembly(bom, pi_global)

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

def update_info(msg):
    return gr.update(value=msg)

def do_generate(prompt):
    if not prompt.strip():
        return gr.update(), update_info("No prompt provided.")
    from src.image_generator import generate_image
    info_update = update_info("Generating image…")
    with gr.Progress(track_tqdm=False) as progress:
        path = generate_image(prompt)
    info_update = update_info(f"Image generated: {path}")
    return gr.update(value=path), info_update

def do_extract(image, prompt_hint):
    if not image:
        return gr.update(visible=False), {}, gr.update(visible=False), gr.update(visible=False), update_info("No image provided.")
    info_update = update_info("Extracting BOM …")
    # Unpack: corrected_bom, warnings
    with gr.Progress(track_tqdm=False) as progress:
        result = get_bom_from_image(image, prompt_hint)
    if isinstance(result, tuple) and len(result) == 2:
        bom, warnings = result
    else:
        bom, warnings = result, []

    df = load_bom_to_df(bom)
    visible = True if df is not None else False
    # Placeholder detection logic
    placeholder = False
    if "head" in bom and "torso" in bom:
        if bom['head'].get('type') == 'sphere' and bom['torso'].get('type') == 'box':
            placeholder = True
    if placeholder:
        info_update = update_info("⚠️ Used placeholder BOM (vision failed)")
    else:
        info_msg = "BOM extracted successfully"
        if warnings:
            info_msg += "\nWarnings:\n" + "\n".join(warnings)
        info_update = update_info(info_msg)
    return gr.update(visible=visible, value=df), bom, gr.update(visible=visible), gr.update(visible=visible), info_update

def do_update_df(df):
    pd = _lazy_import_pandas()
    if not pd or df is None:
        return {}
    bom = df_to_bom(pd.DataFrame(df))
    return bom

def do_skeleton(bom, h, leg, arm):
    param = {"height": h, "leg_length": leg, "arm_length": arm}
    macro = get_skeleton_macro(bom, param)
    path = save_macro_file(macro, "skeleton.FCMacro")
    return gr.update(value=path, visible=True), macro

def do_assembly(bom):
    macro = get_assembly_macro(bom)
    path = save_macro_file(macro, "assembly.FCMacro")
    return gr.update(value=path, visible=True), macro

from src.cache import cached_get_answers
from src.prompts import get_parametric_prompt, get_explanation_prompt
from src.logger import get_logger

logger = get_logger("web_ui")

MODEL_OPTIONS = ["gpt-4", "gpt-3.5-turbo"]

def launch_web_ui():
    if not HAS_GRADIO:
        logger.warning("Gradio not installed; web UI unavailable.")
        return

    def _create_audio_input():
        """
        Helper to instantiate gr.Audio with compatible arguments for Gradio 3.x and 4.x.
        """
        params = inspect.signature(gr.Audio).parameters
        if "source" in params:
            kwargs = {
                "source": "microphone",
                "type": "filepath",
                "label": "🎤 Record",
                "scale": 4,
            }
        else:
            kwargs = {
                "sources": ["microphone"],
                "type": "filepath",
                "label": "🎤 Record",
                "scale": 4,
            }
        return gr.Audio(**kwargs)

    def _create_chatbot():
        """
        Helper to instantiate gr.Chatbot with compatible arguments for Gradio 3.x and 4.x,
        avoiding deprecation warnings for the 'label' and 'type' parameters.
        """
        params = inspect.signature(gr.Chatbot).parameters
        if "type" in params:
            # Gradio 4.x expects the 'type' parameter
            return gr.Chatbot(label="Query2CAD Conversation", type="tuples")  # Explicitly set type to suppress deprecation warning
        else:
            # Gradio 3.x does not accept 'type'
            return gr.Chatbot(label="Query2CAD Conversation")

    def infer(query, model, parametric, explain):
        prompt = query
        if parametric:
            prompt = get_parametric_prompt(query)
        macro = cached_get_answers(model, prompt)
        explanation = ""
        if explain:
            explanation = cached_get_answers(model, get_explanation_prompt(macro))
        return macro, explanation

    def chat_send(user_message, chat_history, model):
        if not user_message.strip():
            return gr.update(), chat_history
        chat_history = chat_history or []
        chat_history.append((user_message, None))
        prompt_parts = []
        for u, a in chat_history[:-1]:
            prompt_parts.append(f"User: {u}\nAI: {a if a is not None else ''}")
        prompt_parts.append(f"User: {user_message}\nAI:")
        prompt = "\n".join(prompt_parts)
        assistant_response = cached_get_answers(model, prompt)
        chat_history[-1] = (user_message, assistant_response)
        return chat_history, chat_history

    def chat_clear():
        return [], []

    with gr.Blocks() as demo:
        gr.Markdown("# Query2CAD Web Interface")

        # ==== 1. SETTINGS PANEL ====
        with gr.Accordion(label="⚙ Settings", open=False):
            openai_key_tb = gr.Textbox(
                label="OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
            )
            together_key_tb = gr.Textbox(
                label="Together API Key",
                value=os.getenv("TOGETHER_API_KEY", ""),
                type="password",
            )
            prompt_max_slider = gr.Slider(
                minimum=500,
                maximum=4000,
                step=100,
                value=int(os.getenv("OPENAI_IMAGE_PROMPT_MAX_CHARS", 4000)),
                label="Max Image Prompt Length",
            )
            save_settings_btn = gr.Button("Save settings")
            settings_msg = gr.Markdown(visible=False)

        def save_settings(openai_key, together_key, prompt_max):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if together_key:
                os.environ["TOGETHER_API_KEY"] = together_key
            os.environ["OPENAI_IMAGE_PROMPT_MAX_CHARS"] = str(int(prompt_max))
            return gr.update(value="✅ Settings saved", visible=True)
        save_settings_btn.click(
            save_settings,
            [openai_key_tb, together_key_tb, prompt_max_slider],
            [settings_msg],
        )

        # Section 1 – Humanoid Robot Pipeline
        gr.Markdown("## Humanoid Robot Pipeline")
        with gr.Row():
            with gr.Column():
                concept_prompt = gr.Textbox(label="Concept description (optional)", lines=2, value="")
                gen_image_btn = gr.Button("Generate Image")
                image_input = gr.Image(type="filepath", label="Upload Robot Image")
                prompt_hint = gr.Textbox(label="Prompt hint (optional)", value="")

                # ==== 2. IMAGE HISTORY GALLERY ====
                def load_history():
                    metadata_path = "results/images/metadata.jsonl"
                    if not os.path.exists(metadata_path):
                        return []
                    images = []
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    lines = lines[-20:]  # take last 20
                    for ln in lines:
                        try:
                            meta = json.loads(ln)
                            path = meta.get("path") or meta.get("filepath") or meta.get("file") or ""
                            prompt = meta.get("prompt", "")
                            if path and os.path.exists(path):
                                images.append([path, prompt])
                        except Exception:
                            continue
                    return images

                def refresh_history():
                    return load_history()

                refresh_history_btn = gr.Button("🔄 Refresh History")
                history_gallery = gr.Gallery(
                    label="Image History",
                    visible=True,
                    height=220,
                    columns=4
                )

                refresh_history_btn.click(
                    refresh_history,
                    inputs=None,
                    outputs=[history_gallery],
                )

                def select_history(evt, gallery, image_comp):
                    idx = getattr(evt, "index", None)
                    if idx is None or not gallery or idx >= len(gallery):
                        return gr.update()
                    path = gallery[idx][0]
                    return gr.update(value=path)

                history_gallery.select(
                    select_history,
                    [history_gallery],
                    [image_input],
                )

                extract_btn = gr.Button("Extract BOM")
            with gr.Column():
                bom_df = gr.Dataframe(
                    label="Editable BOM",
                    headers=["type", "material", "component", "side"],
                    column_config={
                        "side": gr.ColumnDropdown(choices=["left", "right", "center", ""])
                    },
                    interactive=True,
                    visible=False
                )
                height_slider = gr.Slider(100, 250, value=180, label="Height (cm)")
                leg_slider = gr.Slider(30, 120, value=70, label="Leg length (cm)")
                arm_slider = gr.Slider(20, 100, value=60, label="Arm length (cm)")
                skeleton_btn = gr.Button("Generate skeleton", visible=False)
                assembly_btn = gr.Button("Build assembly", visible=False)
                macro_download = gr.File(label="Download Macro", visible=False)
                feedback = gr.Button("👍", visible=True)
        info_box = gr.Markdown("Status", label="Status", visible=True)
        state_bom = gr.State({})
        state_macro = gr.State("")

        # Hook text→image generation button
        gen_image_btn.click(
            do_generate,
            inputs=[concept_prompt],
            outputs=[image_input, info_box],
        )
        extract_btn.click(
            do_extract,
            [image_input, prompt_hint],
            [bom_df, state_bom, skeleton_btn, assembly_btn, info_box]
        )
        bom_df.change(do_update_df, [bom_df], [state_bom])
        skeleton_btn.click(
            do_skeleton,
            [state_bom, height_slider, leg_slider, arm_slider],
            [macro_download, state_macro]
        )
        assembly_btn.click(lambda bom: do_assembly(bom), [state_bom], [macro_download, state_macro])
        feedback.click(lambda: print("Feedback: thumbs up!"), None, None)

        # Section 2 – Macro Generator
        gr.Markdown("## Macro Generator")
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
            retriever.add_example(query, macro)
            return None

        def feedback_bad(query, macro):
            from src.retrieval import Retriever
            retriever = Retriever()
            retriever.add_negative_example(query, "User marked as bad result.")
            return None

        run_btn.click(
            fn=infer,
            inputs=[query, model, parametric, explain],
            outputs=[macro_out, explanation_out]
        )
        thumbs_up.click(fn=feedback_good, inputs=[query, macro_out], outputs=None)
        thumbs_down.click(fn=feedback_bad, inputs=[query, macro_out], outputs=None)

        # Section 3 – Chat with Query2CAD AI
        gr.Markdown("## Chat with Query2CAD AI")
        chat_model = gr.Radio(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model")
        chatbot = _create_chatbot()
        chat_state = gr.State([])

        audio_components_visible = HAS_OPENAI
        audio_row = None
        audio_in = None
        send_audio_btn = None
        audio_warning_box = None

        if audio_components_visible:
            with gr.Row():
                audio_in = _create_audio_input()
                send_audio_btn = gr.Button("Send Audio", scale=1)
            audio_warning_box = gr.Markdown("", visible=False)
        else:
            audio_warning_box = gr.Markdown(
                "⚠️ Voice input requires the `openai` Python package. Install with `pip install openai`.",
                visible=True,
            )

        with gr.Row():
            user_message = gr.Textbox(
                label=None, 
                placeholder="Ask Query2CAD AI...",
                lines=2,
                scale=6,
            )
            send_btn = gr.Button("Send", scale=1)
            clear_btn = gr.Button("Clear", scale=1)
            export_btn = gr.Button("Export History", scale=1)

        export_file = gr.File(label="Download Chat History (JSON)", visible=False, interactive=True, file_types=[".json"])

        def transcribe_and_send(audio_filepath, chat_state, chat_model):
            if not HAS_OPENAI:
                return gr.update(), chat_state, gr.update(value="⚠️ openai package not installed.", visible=True)
            if not audio_filepath:
                return gr.update(), chat_state, gr.update(value="⚠️ Please record audio before sending.", visible=True)
            api_key = os.environ.get("OPENAI_API_KEY", None)
            if not api_key:
                return gr.update(), chat_state, gr.update(value="⚠️ Please set your OPENAI_API_KEY environment variable to enable audio transcription.", visible=True)
            try:
                # Set API key globally for openai>=1.12.0+
                openai.api_key = api_key
                with open(audio_filepath, "rb") as f:
                    transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
                # Prefer transcript.text, fallback to dict["text"] if necessary
                text = getattr(transcript, "text", None)
                if text is None and isinstance(transcript, dict):
                    text = transcript.get("text", "")
                if text is None:
                    text = ""
                if not text.strip():
                    return gr.update(), chat_state, gr.update(value="⚠️ No transcription result.", visible=True)
            except Exception as e:
                return gr.update(), chat_state, gr.update(value=f"⚠️ Transcription failed: {e}", visible=True)
            chat_result, new_state = chat_send(text, chat_state, chat_model)
            return chat_result, new_state, gr.update(value="", visible=False)

        send_btn.click(
            fn=chat_send,
            inputs=[user_message, chat_state, chat_model],
            outputs=[chatbot, chat_state]
        )
        clear_btn.click(
            fn=chat_clear,
            inputs=None,
            outputs=[chatbot, chat_state]
        )
        # Audio send logic
        if audio_components_visible and send_audio_btn and audio_in:
            send_audio_btn.click(
                fn=transcribe_and_send,
                inputs=[audio_in, chat_state, chat_model],
                outputs=[chatbot, chat_state, audio_warning_box]
            )

        def export_chat_history(chat_history):
            if not chat_history or len(chat_history) == 0:
                return gr.update(visible=False, value=None)
            export_dir = os.path.join("results", "chat_history")
            utils.ensure_dir(export_dir)
            data = [
                {"user": u, "assistant": a}
                for (u, a) in chat_history
            ]
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"chat_{now}.json"
            fpath = os.path.join(export_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return gr.update(value=fpath, visible=True)

        export_btn.click(
            fn=export_chat_history,
            inputs=[chat_state],
            outputs=[export_file],
        )

    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))

if __name__ == "__main__":
    if HAS_GRADIO:
        utils.ensure_startup_dirs()
        launch_web_ui()
    else:
        logger = logging.getLogger("web_ui")
        logger.warning("Gradio not installed; skipping web UI launch.")