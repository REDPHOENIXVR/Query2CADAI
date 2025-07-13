─────────────────────────────
## Instant Feedback and Vision BOM

- Giving a 👍 ("thumbs-up") instantly teaches the model: your query and its generated macro are added to the ExampleRetriever for future improvements.  
  > Now, a thumbs-up automatically computes and stores an embedding for your query—no manual setup required. This enables the retriever to learn from your feedback and improves future retrievals.
- If you provide an OpenAI API key (or OpenRouter), Vision→BOM extraction uses the real OpenAI Vision endpoint (not a dummy) and validates the result. Otherwise, a placeholder/dummy BOM is used.
  You can override the vision model via the `OPENAI_VISION_MODEL` env var (defaults to `gpt-4o`).

# Query2CAD

...

# Humanoid Robot Workflow

This workflow takes you from a robot image to a FreeCAD macro for assembly or skeleton simulation.

## Step-by-Step Usage

1. **Startup**  
   Ensure required folders:  
   - `library/parts/` : Place STEP/FCStd files and their YAML sidecars here  
   - `library/index/` : FAISS index and manifest (auto-generated)  
   - `results/bom/`   : Extracted BOM JSONs  

2. **Web UI**  
   - Launch with:  
     `python -m src.web_ui`  
     By default, this launches the **Chat UI**, which includes both the Chat tab and the original pipeline UI.

   - To launch just the original Humanoid Robot Pipeline UI (without Chat), use:  
     `python -m src.web_ui --mode pipeline`

   **NEW: Text → Image → CAD Flow**

   You can now generate a concept image directly from a text prompt using the “Concept description” box and “Generate Image” button above the image upload. This uses OpenAI's DALL·E (or a compatible model) if available, or produces a placeholder image otherwise (see environment variable `OPENAI_IMAGE_MODEL`). After generating, you may review and proceed to extract the BOM as before.

   **Modes:**  
   - `chat` (default): Rich UI with Chat tab and all features.
   - `pipeline`: Only the original Humanoid Robot Pipeline UI (no Chat).

   **Chat with Query2CAD AI:**  
   The Web UI now includes a **Chat** tab by default, where you can have a free-form conversation with the Query2CAD AI assistant. You can type questions or requests in natural language, click **Send**, and receive AI responses in a conversational format. Use the **Clear** button to reset the chat history at any time. You can select the backend AI model using the radio button above the chat.  
   You can also export your chat history at any time via the **Export History** button, which will generate a downloadable JSON file containing your conversation.

   **NEW: Voice/Microphone Input**

   The Chat tab supports voice input via your microphone. Click the 🎤 **Record** button, record your question, then click **Send Audio**. Your voice will be transcribed to text using OpenAI Whisper and sent as a message.  
   > **Note:** This feature requires both the `openai` Python package and a valid `OPENAI_API_KEY` environment variable. If either is missing, the microphone input will be disabled.

   This feature is useful for exploratory design, asking clarifying questions, or just experimenting with Query2CAD's capabilities interactively.

   - Upload a photo of a robot (or concept drawing).
   - Optionally provide a prompt hint.
   - Click **Extract BOM**:  
     The system extracts a Bill of Materials (BOM) from the image using Vision AI (if available), or returns an example for testing.
   - The editable BOM appears: adjust as needed.
   - Click **Generate skeleton** to produce a FreeCAD macro with placeholder geometry for limbs.
   - Click **Build assembly** to generate a macro that will import matching STEP parts and assemble them.
   - Download the macro and run in FreeCAD.

3. **Assembly/Testing**  
   - Use the macro in FreeCAD to visualize or modify the design.
   - For collision testing, use the `src/collision.py` module (requires pybullet).

**NOTE:** After adding new STEP files, rebuild the parts index:
```bash
python -m src.build_parts_index
```

## Robot Pipeline CLI

You can now run the full humanoid pipeline with a single command:

```bash
python -m src.robot_pipeline --image path/to/input.png --outdir results/run1
```

Or, to generate an image from a text prompt and use it as input:

```bash
python -m src.robot_pipeline --text "your concept prompt" --outdir results/run2
```

If both `--image` and `--text` are supplied, the explicit image is used and `--text` is ignored (with a note). The image is generated using OpenAI (DALL·E by default, override with `OPENAI_IMAGE_MODEL`), or a placeholder if unavailable.

This will:
- Extract a BOM from the image
- Generate a FreeCAD skeleton macro
- Build or update the parts index
- Build an assembly macro

Outputs will be in the specified `--outdir`.

## File Structure

- `src/parts_index.py` : Parts metadata parse/index/query
- `src/vision_bom.py`  : Image → BOM extractor
- `src/skeleton.py`    : Skeleton macro generator
- `src/assembly_builder.py` : Assembly macro builder
- `src/collision.py`   : STEP collision checker

## Library maintenance

You can now scan your parts library for missing metadata and auto-generate YAML stubs for new or incomplete parts:

```bash
# Scan for missing metadata and generate stub YAMLs
python -m src.parts_validator  # or --ai to auto-fill using GPT
```

- This command will create YAML stubs for any CAD files missing them, and flag any parts lacking essential metadata (like model or category).
- A report will be printed and written to `library/index/flagged_parts.txt` listing all parts needing review.
- Optionally, use `--ai` to fill in metadata using OpenAI if `openai` is installed and `OPENAI_API_KEY` is set.

## Requirements

- Python 3.8+
- pyyaml, sentence-transformers, faiss-cpu, pybullet, pandas, gradio, python-multipart

# Adding the HOPEJr Robot Library

To enrich your Query2CAD part library with a set of high-quality humanoid robot parts, you can import the open-source [HOPEJr](https://github.com/TheRobotStudio/HOPEJr) repository.

## 1. Add the HOPEJr repository as a submodule

```bash
git submodule add https://github.com/TheRobotStudio/HOPEJr external/HOPEJr
```

## 2. Import the STEP files into Query2CAD

```bash
python -m src.import_hopejr
```

This will copy all STEP files from the HOPEJr repository into `library/parts/` (with a `hopejr_` prefix) and create minimal YAML metadata for each part.

## 3. Rebuild the parts index

```bash
python -m src.build_parts_index
```

## 4. Licensing

HOPEJr is licensed under [GPL-3.0](https://github.com/TheRobotStudio/HOPEJr/blob/master/LICENSE). Please review their repository and license before commercial or derivative use.