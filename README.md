# HeartMuse - AI Music Generator with Smart Lyrics

**HeartMuse** is an intuitive web-based interface for creating high-quality AI-generated music **completely locally** on your machine. It combines the power of **HeartMuLa** (state-of-the-art open-source music generation model) with intelligent lyrics generation using local LLMs, giving you complete creative control without relying on cloud services.

![HeartMuse Interface](docs/screenshot.png)

## Get Started in 2 Minutes

```bash
# 1. Clone and install
git clone https://github.com/strnad/heartmuse.git
cd heartmuse
./install.sh      # Linux/macOS
# or: install.bat   # Windows

# 2. Run
./run.sh          # Linux/macOS
# or: run.bat       # Windows
```

Open **http://localhost:7860** and start creating!

> **That's it!** The installer automatically creates a virtual environment, clones the HeartMuLa library, and installs all dependencies. AI models download automatically on first generation.

---

## Features at a Glance

- **Smart Text Generation** - AI writes lyrics, titles, tags, and descriptions via Ollama (local) or OpenAI
- **HeartMuLa Music Generation** - two model variants (RL and Base), up to 240s songs
- **Style Transfer** - upload reference audio to influence the musical style (MuQ-MuLan)
- **Audio Transcription** - extract lyrics from existing audio (HeartTranscriptor)
- **Batch Variants** - generate 1-10 variations of the same song in one go
- **Edit Instructions** - refine generated text with natural language commands
- **History & Playlist** - browse, replay, and manage all past generations
- **Live Memory Monitor** - real-time GPU/RAM usage tracking
- **Seed Control** - reproduce exact results with fixed seeds
- **100% Local** - everything runs on your machine with Ollama (or use OpenAI API)

---

## Why HeartMuse?

### Flexible Text Generation - Total Creative Control

HeartMuse features a **modular, field-by-field generation system** that adapts to your creative workflow. Every field has its own "Generate/Enhance" checkbox, giving you granular control over what AI generates and what you write yourself.

#### Four Independent Fields, Endless Combinations

| Field | What It Does | Generate/Enhance Checkbox |
|-------|-------------|---------------------------|
| **Description** | Your creative brief for the song | AI can expand vague ideas into detailed descriptions |
| **Title** | Song name | AI suggests catchy, thematic titles |
| **Lyrics** | Full song text with sections | AI writes/extends verses, choruses, bridges |
| **Tags** | Music production tags for HeartMuLa | AI suggests genre, mood, instruments, tempo |

**Each checkbox is independent** - mix and match any combination:
- All four checked - AI generates everything from scratch
- Only lyrics checked - AI writes lyrics, you control title and tags
- Title + tags checked - You write lyrics, AI handles metadata
- Nothing checked - Use exactly what you entered, no AI changes

#### Context-Aware Intelligence

The magic happens when you **provide partial content**:

```
+----------------------------------+--------------------------------+
| You provide:                     | AI understands:                |
+----------------------------------+--------------------------------+
| Description: "upbeat summer"     | -> Uses as creative direction  |
| Title: (empty)                   | -> Will generate               |
| Lyrics: "Feel the sunshine"      | -> Context for generation      |
| Tags: (empty)                    | -> Will generate               |
|                                                                   |
| Checkboxes: [x] Title  [x] Tags  [ ] Lyrics                       |
|                                                                   |
| Result: AI generates title and tags that match YOUR lyrics        |
|         and description. Your lyrics stay untouched.              |
+-------------------------------------------------------------------+
```

**Unchecked fields with values become context** - AI reads them but won't modify them. This ensures coherent results that respect your creative input.

#### Smart Lyrics Preservation

When extending existing lyrics, HeartMuse offers **two levels of protection**:

| Syntax | Protection Level | Use Case |
|--------|-----------------|----------|
| `"exact text"` | **100% locked** - never changed, character-for-character | Your signature lines, hooks |
| `regular text` | **Preserved** - meaning kept, minor polish allowed | Draft verses you want improved |

**Example:**
```
[chorus]
"This exact line will never change!"
This line might get slight improvements
```

Result: The quoted line is untouchable. The unquoted line may be refined while keeping its meaning.

#### Extend, Don't Replace

When "Generate/Enhance" is checked for lyrics that already have content:
- AI **adds** new sections (verses, bridge, outro)
- AI **completes** unfinished sections
- AI **preserves** everything you wrote
- AI **never deletes** your content
- AI **never rewrites** existing sections from scratch

#### Edit Instructions

Use the **Edit Instructions** field to give natural language commands for refining generated content:

```
Examples:
- "Change the name Eva to Ela"
- "Add two more verses"
- "Rework the chorus to be more upbeat"
- "Replace all references to rain with sunshine"
```

The AI applies your edits while preserving the rest of the content. Quoted text remains protected even during edits.

#### Duration-Aware Lyrics

Set your target song length, and AI adjusts lyrics accordingly:

| Duration | AI Behavior |
|----------|-------------|
| Under 60s | Short and punchy - 1-2 sections |
| 60-120s | Standard structure - verse, chorus, verse |
| Over 120s | Extended - multiple verses, bridge, outro |

#### From-Scratch Creative Mode

Leave all fields empty and click "Generate Text" - AI creates a completely original song concept:
- Invents a unique theme and story
- Writes complete lyrics with proper structure
- Suggests a fitting title
- Recommends appropriate musical tags

**Each generation is different** - use this for inspiration or when you want to be surprised!

---

### Real-World Workflow Examples

#### Example 1: Full AI Generation
```
Input:   (everything empty, all checkboxes checked)
Output:  Complete original song - title, lyrics, tags, ready for music generation
```

#### Example 2: Your Lyrics, AI Metadata
```
Input:   Your complete lyrics in the Lyrics field
         [ ] Description  [ ] Lyrics  [x] Title  [x] Tags
Output:  AI suggests a perfect title and musical tags that match YOUR lyrics
```

#### Example 3: Expand a Hook
```
Input:   Lyrics: "[chorus]\nDancing in the moonlight"
         [x] Lyrics checked
Output:  AI keeps your chorus and adds verses, bridge, outro around it
```

#### Example 4: Protected Refrain + AI Verses
```
Input:   Lyrics:
         [chorus]
         "We are the champions, my friend"
         "And we'll keep on fighting till the end"

         [verse]
         (something about struggle and victory)

         [x] Lyrics checked

Output:  Quoted chorus: 100% unchanged
         Verse placeholder: AI writes full lyrics about struggle and victory
```

#### Example 5: Enhance Vague Description
```
Input:   Description: "sad piano song"
         [x] Description checked
Output:  Description expanded to: "A melancholic piano ballad with introspective
         lyrics about lost love, featuring gentle arpeggios and emotional vocal
         delivery in a minor key"
```

#### Example 6: Iterative Refinement
```
Round 1: Generate title + lyrics from description
Round 2: Use Edit Instructions: "Make the chorus more energetic"
Round 3: Tweak tags manually, generate music
Round 4: Generate 3 batch variants, pick the best one
```

---

### Style Transfer (Experimental)

Upload a **reference audio track** to influence the musical style of your generation. HeartMuse uses **MuQ-MuLan** to extract a style embedding from the reference and inject it into the HeartMuLa generation process.

- **Style Strength** slider (0-10x) - control how strongly the reference influences output
- Runs on **CPU** - no GPU memory impact, works alongside HeartMuLa
- Supports common audio formats (MP3, WAV, FLAC, etc.)

> Works best with clear, well-produced reference tracks. The model captures high-level style characteristics (genre, mood, instrumentation) rather than copying melodies.

---

### Audio Transcription

The **Transcribe** tab lets you extract lyrics from existing audio recordings using **HeartTranscriptor** (Whisper-based model).

- Upload any audio file and get transcribed lyrics
- Click **"Send to Generator"** to use transcribed lyrics as a starting point
- For best results, use **source-separated vocal tracks** (e.g., via [Demucs](https://github.com/facebookresearch/demucs))

> Toggle with `TRANSCRIPTION=true/false` in `.env`

---

### Batch Generation & Reproducibility

- **Batch Variants** (1-10) - generate multiple versions from the same lyrics/tags in one run
- **Seed Control** - set a specific seed to reproduce exact results, or use `-1` for random
- **Post-generation Statistics** - view timing breakdown (text gen, style extraction, music gen per variant), GPU peak VRAM, model variant, and seed value

---

### History & Playlist

The **History** tab keeps all your generations organized:

- **Playlist Player** - sequential or shuffle playback with next/prev controls and seeking
- **History Cards** - title, description, tags, audio player, and generation parameters for each song
- **Actions** - Load to Generator (reuse settings), Load for Edit (reuse settings + seed), Delete
- **Pagination** - browse through all past generations, 10 per page

All generations are stored as MP3 + JSON metadata in the `output/` directory.

---

### 100% Local & Private

- **Ollama backend** - everything runs on your computer, no data leaves your machine
- **Or OpenAI API** - for those who prefer cloud models
- **Switch with one click** - choose your backend based on the situation

### Effortless HeartMuLa Setup

Forget manual model downloads, path configuration, and dependency hell:

```
./install.sh  ->  Done!
```

HeartMuse handles everything:
- Creates an isolated Python environment
- Clones and installs [HeartMuLa library](https://github.com/HeartMuLa/heartlib)
- Installs all dependencies automatically
- Downloads models on first use (from Hugging Face)

---

## How It Works

```
+------------------+     +------------------+     +------------------+     +------------------+
|   Your Input     | --> |   LLM (Ollama/   | --> | Style Extraction | --> |   HeartMuLa      |
|                  |     |   OpenAI)        |     |   (optional)     |     |   Music Gen      |
|  - Description   |     |                  |     |                  |     |                  |
|  - Checkboxes    |     |  Generates:      |     |  MuQ-MuLan:      |     |  Creates:        |
|  - Edit instrs.  |     |  - Title         |     |  Reference audio |     |  - MP3 audio     |
|  - Ref. audio    |     |  - Lyrics        |     |  -> 512D style   |     |  - Up to 240s    |
|                  |     |  - Tags          |     |     embedding    |     |  - 1-10 variants |
+------------------+     +------------------+     +------------------+     +------------------+
```

## Configuration

Create a `.env` file (or copy `.env.example`):

```bash
# LLM Backend (Ollama = local, OpenAI = cloud)
LLM_BACKEND=Ollama

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=glm-4.7-flash

# OpenAI
# OPENAI_API_KEY=sk-...
# OPENAI_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4.1-mini

# LLM generation
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=120

# Music generation
MUSIC_TEMPERATURE=1.0
MUSIC_CFG_SCALE=1.5
MUSIC_TOPK=50
MUSIC_MAX_LENGTH_SEC=240
MUSIC_NUM_VARIANTS=1

# Model variant: rl (recommended) or base
MODEL_VARIANT=rl

# Lazy loading - loads models on demand, frees VRAM between stages
LAZY_LOAD=true

# Features (true/false)
STYLE_TRANSFER=true
TRANSCRIPTION=true

# Server
SERVER_HOST=127.0.0.1
SERVER_PORT=7860
```

All variables are optional - sensible defaults are used when not set.

### Model Variants

| Variant | Description | Config Value |
|---------|-------------|-------------|
| **HeartMuLa 3B RL** (Recommended) | RL-trained, better style and tag adherence | `MODEL_VARIANT=rl` |
| **HeartMuLa 3B** (Base) | Original model | `MODEL_VARIANT=base` |

Switch between variants from the UI dropdown or `.env`. The pipeline automatically reloads when switching.

### Setting Up Ollama (Recommended)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Download a model
ollama pull glm-4.7-flash

# 3. Start (if not running automatically)
ollama serve
```

## Requirements

- **NVIDIA GPU with 16GB+ VRAM** (e.g., RTX 4080, RTX 3090, RTX 4090)
- CPU fallback available (significantly slower)
- Style transfer (MuQ-MuLan) runs on CPU and needs ~2.5 GB RAM additionally

## Memory Management

HeartMuse provides several tools to manage GPU and system memory:

- **Lazy Loading** (default: on) - loads models on demand and frees VRAM between stages
- **Unload Buttons** - individually unload Music Pipeline, MuQ Model, Ollama Model, or Transcriptor
- **Auto-unload Ollama** (default: on) - automatically unloads Ollama model after text generation
- **Live Memory Monitor** - real-time display of GPU utilization, VRAM usage, and RAM usage (updates every 3 seconds)
- **Shorter song length** - reduce max duration to lower peak VRAM usage

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Out of memory" | Reduce song length, enable lazy loading, use "Unload" buttons |
| Ollama not connecting | Verify `ollama serve` is running |
| Models not downloading | Check internet connection and Hugging Face access |
| Style transfer slow | Normal - MuQ-MuLan runs on CPU (~2.5 GB RAM) |
| Transcription inaccurate | Use source-separated vocal tracks (e.g., Demucs) |

## Acknowledgments

HeartMuse builds on the excellent work by:
- [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) - Music generation model and HeartCodec audio codec
- [OpenMuQ/MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) - Music-text alignment model for style transfer
- [HeartTranscriptor](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss) - Audio transcription model

## Support the Project

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub-ff69b4?style=for-the-badge&logo=github)](https://github.com/sponsors/strnad)

**Bitcoin:** `bc1qgsn45g02wran4lph5gsyqtk0k7t98zsg6qur0y`

## License

MIT License - see [LICENSE](LICENSE)

The HeartMuLa library has its own license - refer to the [HeartMuLa repository](https://github.com/HeartMuLa/heartlib).

---

**Made with HeartMuLa | Developed with assistance from [Claude Code](https://claude.ai/code)**

*Create music with AI, own your creativity*
