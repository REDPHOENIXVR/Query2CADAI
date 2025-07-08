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
import logging
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
    # For lazy loading new modules
    import importlib
    return importlib

# Set up global PartIndex cache
pi_global = PartIndex.load()
if pi_global.index is None or len(pi_global.parts) == 0:
    pi_global.build_index("library/parts")

# New: Import robot pipeline modules lazily
def get_bom_from_image(image_bytes, prompt_hint=""):
    import src.vision_bom as vision_bom
    return vision_bom.extract_bom(image_bytes, prompt_hint)

def get_skeleton_macro(bom):
    import src.skeleton as skeleton
    return skeleton.generate_skeleton(bom, None)

def get_assembly_macro(bom):
    # build assembly using cached global PartIndex
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
                image_input = gr.Image(type="filepath", label="Upload Robot Image")
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
                return {}
            bom = df_to_bom(pd.DataFrame(df))
            return bom

        def do_skeleton(bom):
            macro = get_skeleton_macro(bom)
            path = save_macro_file(macro, "skeleton.FCMacro")
            return gr.update(value=path, visible=True), macro

        def do_assembly(bom):
            macro = get_assembly_macro(bom)
            path = save_macro_file(macro, "assembly.FCMacro")
            return gr.update(value=path, visible=True), macro

        extract_btn.click(do_extract, [image_input, prompt_hint], [bom_df, state_bom, skeleton_btn, assembly_btn])
        bom_df.change(do_update_df, [bom_df], [state_bom])
        skeleton_btn.click(lambda bom: do_skeleton(bom), [state_bom], [macro_download, state_macro])
        assembly_btn.click(lambda bom: do_assembly(bom), [state_bom], [macro_download, state_macro])

        # Feedback button just log for now
        feedback.click(lambda: print("Feedback: thumbs up!"), None, None)

    return demo

import argparse

# ... (rest of code remains unchanged above)

# Place this block at the very end of the file, after all function definitions (including launch_web_ui).

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query2CAD Web UI Launcher")
    parser.add_argument(
        "--mode",
        choices=["chat", "pipeline"],
        default="chat",
        help="Which UI to launch: 'chat' (default, with Chat tab) or 'pipeline' (Humanoid Robot Pipeline only)."
    )
    args = parser.parse_args()

    if HAS_GRADIO:
        utils.ensure_startup_dirs()
        if args.mode == "pipeline":
            ui_main().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
        else:
            launch_web_ui()
    else:
        logger = logging.getLogger("web_ui")
        logger.warning("Gradio not installed; skipping web UI launch.")

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

    def chat_send(user_message, chat_history, model):
        # Append user message (assistant response = None for now)
        if not user_message.strip():
            return gr.update(), chat_history  # Don't submit empty
        chat_history = chat_history or []
        chat_history.append((user_message, None))
        # Build prompt from history
        prompt_parts = []
        for u, a in chat_history[:-1]:
            prompt_parts.append(f"User: {u}\nAI: {a if a is not None else ''}")
        # Add latest message, with AI: as last line
        prompt_parts.append(f"User: {user_message}\nAI:")
        prompt = "\n".join(prompt_parts)
        # Call model
        assistant_response = cached_get_answers(model, prompt)
        # Update last tuple with response
        chat_history[-1] = (user_message, assistant_response)
        return chat_history, chat_history

    def chat_clear():
        return [], []

    with gr.Blocks() as demo:
        gr.Markdown("# Query2CAD Web UI")
        with gr.Tabs():
            with gr.Tab("Main"):
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
                    # No embedding parameter needed; handled internally
                    retriever.add_example(query, macro)
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

            with gr.Tab("Chat"):
                gr.Markdown("### Chat with Query2CAD AI")
                # Use the same model selection as in Main tab
                chat_model = gr.Radio(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model")
                chatbot = gr.Chatbot(label="Query2CAD Conversation")
                chat_state = gr.State([])  # List of (user, assistant) tuples

                # --- Voice input section ---
                # Only show if openai is installed
                audio_components_visible = HAS_OPENAI

                audio_row = None
                audio_in = None
                send_audio_btn = None
                audio_warning_box = None

                if audio_components_visible:
                    with gr.Row():
                        audio_in = gr.Audio(source="microphone", type="filepath", label="🎤 Record", scale=4)
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

                # --- Audio-to-text and send logic ---
                def transcribe_and_send(audio_filepath, chat_state, chat_model):
                    # Hide warning box by default
                    if not HAS_OPENAI:
                        return gr.update(), chat_state, gr.update(value="⚠️ openai package not installed.", visible=True)
                    if not audio_filepath:
                        # No audio file to transcribe
                        return gr.update(), chat_state, gr.update(value="⚠️ Please record audio before sending.", visible=True)
                    api_key = os.environ.get("OPENAI_API_KEY", None)
                    if not api_key:
                        return gr.update(), chat_state, gr.update(value="⚠️ Please set your OPENAI_API_KEY environment variable to enable audio transcription.", visible=True)
                    try:
                        with open(audio_filepath, "rb") as f:
                            transcript = openai.audio.transcriptions.transcribe("whisper-1", f, api_key=api_key)
                        if isinstance(transcript, dict):
                            text = transcript.get("text", "")
                        else:
                            text = transcript
                        if not text.strip():
                            return gr.update(), chat_state, gr.update(value="⚠️ No transcription result.", visible=True)
                    except Exception as e:
                        return gr.update(), chat_state, gr.update(value=f"⚠️ Transcription failed: {e}", visible=True)
                    # Use chat_send to continue as user message
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

                # Audio send logic: already connected above if HAS_OPENAI

                # Export chat history (keep only one definition)
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
    parser = argparse.ArgumentParser(description="Query2CAD Web UI Launcher")
    parser.add_argument(
        "--mode",
        choices=["chat", "pipeline"],
        default="chat",
        help="Which UI to launch: 'chat' (default, with Chat tab) or 'pipeline' (Humanoid Robot Pipeline only)."
    )
    args = parser.parse_args()

    if HAS_GRADIO:
        utils.ensure_startup_dirs()
        if args.mode == "pipeline":
            ui_main().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
        else:
            launch_web_ui()
    else:
        logger = logging.getLogger("web_ui")
        logger.warning("Gradio not installed; skipping web UI launch.")
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

    def chat_send(user_message, chat_history, model):
        # Append user message (assistant response = None for now)
        if not user_message.strip():
            return gr.update(), chat_history  # Don't submit empty
        chat_history = chat_history or []
        chat_history.append((user_message, None))
        # Build prompt from history
        prompt_parts = []
        for u, a in chat_history[:-1]:
            prompt_parts.append(f"User: {u}\nAI: {a if a is not None else ''}")
        # Add latest message, with AI: as last line
        prompt_parts.append(f"User: {user_message}\nAI:")
        prompt = "\n".join(prompt_parts)
        # Call model
        assistant_response = cached_get_answers(model, prompt)
        # Update last tuple with response
        chat_history[-1] = (user_message, assistant_response)
        return chat_history, chat_history

    def chat_clear():
        return [], []

    with gr.Blocks() as demo:
        gr.Markdown("# Query2CAD Web UI")
        with gr.Tabs():
            with gr.Tab("Main"):
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
                    # No embedding parameter needed; handled internally
                    retriever.add_example(query, macro)
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

            with gr.Tab("Chat"):
                gr.Markdown("### Chat with Query2CAD AI")
                # Use the same model selection as in Main tab
                chat_model = gr.Radio(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model")
                chatbot = gr.Chatbot(label="Query2CAD Conversation")
                chat_state = gr.State([])  # List of (user, assistant) tuples

                # --- Voice input section ---
                # Only show if openai is installed
                audio_components_visible = HAS_OPENAI

                audio_row = None
                audio_in = None
                send_audio_btn = None
                audio_warning_box = None

                if audio_components_visible:
                    with gr.Row():
                        audio_in = gr.Audio(source="microphone", type="filepath", label="🎤 Record", scale=4)
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

                # --- Audio-to-text and send logic ---
                def transcribe_and_send(audio_filepath, chat_state, chat_model):
                    # Hide warning box by default
                    if not HAS_OPENAI:
                        return gr.update(), chat_state, gr.update(value="⚠️ openai package not installed.", visible=True)
                    if not audio_filepath:
                        # No audio file to transcribe
                        return gr.update(), chat_state, gr.update(value="⚠️ Please record audio before sending.", visible=True)
                    api_key = os.environ.get("OPENAI_API_KEY", None)
                    if not api_key:
                        return gr.update(), chat_state, gr.update(value="⚠️ Please set your OPENAI_API_KEY environment variable to enable audio transcription.", visible=True)
                    try:
                        with open(audio_filepath, "rb") as f:
                            transcript = openai.audio.transcriptions.transcribe("whisper-1", f, api_key=api_key)
                        if isinstance(transcript, dict):
                            text = transcript.get("text", "")
                        else:
                            text = transcript
                        if not text.strip():
                            return gr.update(), chat_state, gr.update(value="⚠️ No transcription result.", visible=True)
                    except Exception as e:
                        return gr.update(), chat_state, gr.update(value=f"⚠️ Transcription failed: {e}", visible=True)
                    # Use chat_send to continue as user message
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

                # Audio send logic: already connected above if HAS_OPENAI

                # Export chat history (keep only one definition)
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