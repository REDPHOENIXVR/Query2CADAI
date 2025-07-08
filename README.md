─────────────────────────────
## Instant Feedback and Vision BOM

- Giving a 👍 ("thumbs-up") instantly teaches the model: your query and its generated macro are added to the ExampleRetriever for future improvements.
- If you provide an OpenAI API key (or OpenRouter), Vision→BOM extraction uses the real OpenAI Vision endpoint (not a dummy) and validates the result. Otherwise, a placeholder/dummy BOM is used.

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

   **Chat with Query2CAD AI:**  
   The Web UI now includes a **Chat** tab, where you can have a free-form conversation with the Query2CAD AI assistant. You can type questions or requests in natural language, click **Send**, and receive AI responses in a conversational format. Use the **Clear** button to reset the chat history at any time. You can select the backend AI model using the radio button above the chat.  
   You can also export your chat history at any time via the **Export History** button, which will generate a downloadable JSON file containing your conversation.
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

## Requirements

- Python 3.8+
- pyyaml, sentence-transformers, faiss-cpu, pybullet, pandas, gradio, python-multipart