# 🎵 HeartMuse - AI Music Generator with Smart Lyrics

**HeartMuse** is an intuitive web-based interface for creating high-quality AI-generated music **completely locally** on your machine. It combines the power of **HeartMuLa** (state-of-the-art open-source music generation model for local inference) with intelligent lyrics generation using local LLMs, giving you complete creative control without relying on cloud services.

## ✨ What Makes HeartMuse Special?

While [HeartMuLa](https://github.com/HeartMuLa/heartlib) provides state-of-the-art music generation capabilities, **HeartMuse** extends it with:

- 🎨 **User-Friendly Web Interface** - No command-line expertise needed
- 📝 **Smart Lyrics Generation** - Leverages local Ollama models or OpenAI API to automatically generate coherent, themed lyrics from simple descriptions
- 🏷️ **Intelligent Tagging** - Automatically generates appropriate music style tags
- 💾 **Complete Privacy** - Run 100% locally with Ollama (no data leaves your machine)
- 📚 **Generation History** - Browse, replay, and manage all your previous creations
- ⚙️ **Flexible Configuration** - Easy-to-use controls for fine-tuning generation parameters

## 🎯 Features

### Smart Text Generation
- **Describe Your Vision**: Simply write what kind of song you want (e.g., "upbeat pop song about summer adventures")
- **Automatic Lyrics**: AI generates full lyrics matching your description and chosen theme
- **Song Titles**: Creative, relevant titles generated automatically
- **Style Tags**: Intelligent tagging system for music genre, mood, and instrumentation

### Powerful Music Generation
- **HeartMuLa 3B Model**: State-of-the-art open-source model for local music generation (3 billion parameters, RL-trained)
- **High-Fidelity Audio**: Uses HeartCodec for superior audio quality
- **Customizable Parameters**: Control temperature, CFG scale, Top-K sampling, and duration
- **GPU Acceleration**: CUDA support with efficient memory management and lazy loading (reduces VRAM usage)
- **Memory Efficient**: Lazy loading feature allows generation on GPUs with limited VRAM

### Dual LLM Backend Support
- **Ollama** (Recommended): Run completely locally with models like `glm-4.7-flash`, `llama3`, `mistral`, etc.
- **OpenAI API**: Use GPT-4o, GPT-4o-mini, or other OpenAI models for lyrics generation

### Seamless Workflow
1. Enter a song description
2. Let AI generate lyrics, title, and tags (or write your own)
3. Click "Generate Music" and get professional-quality audio
4. Browse your creation history anytime

## 🚀 Quick Start

### Prerequisites
- **Git** - For cloning repositories and submodules
- **Python 3.10+**
- **CUDA-compatible GPU** (recommended **12GB VRAM** (8GB VRAM minimum) for HeartMuLa-3B model)
- **Ollama** (optional, for local lyrics generation) - [Download Ollama](https://ollama.ai)

### Installation

**Linux / macOS:**
```bash
git clone https://github.com/yourusername/heartmuse.git
cd heartmuse
./install.sh
```

**Windows:**
```bash
git clone https://github.com/yourusername/heartmuse.git
cd heartmuse
install.bat
```

The installer will:
- Create a Python virtual environment
- Clone the HeartMuLa library
- Install all dependencies
- Prepare your system for music generation

### Running HeartMuse

**Linux / macOS:**
```bash
./run.sh
```

**Windows:**
```bash
run.bat
```

Open your browser to **http://localhost:7860** and start creating!

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Choose your LLM backend
LLM_BACKEND=Ollama          # or OpenAI

# Ollama Configuration (Local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=glm-4.7-flash

# OpenAI Configuration (API)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Music Generation Parameters
MUSIC_TEMPERATURE=1.0       # Creativity (0.5-2.0)
MUSIC_CFG_SCALE=1.5        # Adherence to prompt (1.0-3.0)
MUSIC_TOPK=50              # Sampling diversity (1-100)
MUSIC_MAX_LENGTH_SEC=240   # Max duration in seconds
```

### Using Ollama (100% Local)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Download a model: `ollama pull glm-4.7-flash` (or `llama3`, `mistral`, etc.)
3. Make sure Ollama is running: `ollama serve`
4. Set `LLM_BACKEND=Ollama` in your `.env`

### Using OpenAI API

1. Get your API key from [platform.openai.com](https://platform.openai.com)
2. Set `OPENAI_API_KEY` in your `.env`
3. Set `LLM_BACKEND=OpenAI`

## 📖 How It Works

HeartMuse orchestrates a two-stage generation pipeline:

### Stage 1: Text Generation (LLM)
- Takes your song description
- Generates contextually appropriate lyrics
- Creates a catchy title
- Suggests music style tags (genre, mood, instruments)

### Stage 2: Music Generation (HeartMuLa)
- Processes lyrics and tags through HeartMuLa's 3B parameter model
- Generates high-fidelity audio using HeartCodec
- Saves output with complete metadata

All generations are saved to the `output/` directory with JSON metadata, making it easy to track your creative journey.

## 🎓 Examples

### Example 1: Upbeat Pop Song
**Description**: "Energetic pop song about chasing dreams"

**Generated Output**:
- **Title**: "Dreams in Motion"
- **Lyrics**: Full verses and chorus about ambition and perseverance
- **Tags**: `pop, upbeat, energetic, electronic, synthesizer`
- **Audio**: 2-3 minute high-quality music track

### Example 2: Melancholic Ballad
**Description**: "Slow, emotional ballad about lost love"

**Generated Output**:
- **Title**: "Fading Echoes"
- **Lyrics**: Heartfelt verses about memories and longing
- **Tags**: `ballad, slow, melancholic, piano, emotional`
- **Audio**: Emotive instrumental with appropriate pacing

## 🙏 Credits & Acknowledgments

HeartMuse is built on top of the incredible work by the **HeartMuLa** team:

- **HeartMuLa Project**: [github.com/HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib)
- **Models**: HeartMuLa-RL-oss-3B (state-of-the-art for local music generation), HeartCodec-oss
- **Research Papers**: Check the [HeartMuLa repository](https://github.com/HeartMuLa/heartlib) for technical details

**Huge thanks to the HeartMuLa authors** for creating and open-sourcing their state-of-the-art music generation technology, making professional-quality AI music generation accessible to everyone for local inference!

## 🛠️ Technology Stack

- **[HeartMuLa](https://github.com/HeartMuLa/heartlib)** - 3B parameter music generation model
- **[Gradio](https://gradio.app)** - Web interface framework
- **[Ollama](https://ollama.ai)** - Local LLM inference
- **[OpenAI API](https://openai.com)** - Cloud LLM option
- **PyTorch** - Deep learning backend
- **Python 3.10+** - Core runtime

## 📋 System Requirements

**Minimum (CPU mode)**:
- Git
- Python 3.10+
- 16GB RAM
- 10GB disk space
- ⚠️ **Note**: CPU mode is functional but significantly slower than GPU

**Recommended (GPU mode)**:
- Git
- CUDA-compatible GPU with **8GB+ VRAM** (e.g., RTX 3070, RTX 4060, or better)
- 16GB system RAM
- 20GB disk space (for models and generated audio)
- CUDA 11.8+ / 12.x

**Memory Optimization**:
- Lazy loading is enabled by default (reduces VRAM footprint)
- Manual "Unload Model" button frees GPU memory between generations
- For GPUs with less VRAM, reduce `MUSIC_MAX_LENGTH_SEC` to generate shorter clips

## 🐛 Troubleshooting

### Models Not Downloading
The first run automatically downloads ~3GB of model weights from Hugging Face. Ensure you have:
- Stable internet connection
- Sufficient disk space in the `ckpt/` directory

### Out of Memory Errors
- Use the "Unload Model" button between generations
- Reduce `MUSIC_MAX_LENGTH_SEC` in GUI or `.env`

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_URL` matches your Ollama installation
- Verify the model is downloaded: `ollama list`

## 💖 Support the Project

If HeartMuse saves you time or helps you create something cool, consider supporting development 🙏

### Sponsor via GitHub
[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub-ff69b4?style=for-the-badge&logo=github)](https://github.com/sponsors/strnad)

### Donate with Bitcoin
`bc1qgsn45g02wran4lph5gsyqtk0k7t98zsg6qur0y`

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The HeartMuLa library has its own license - please refer to the [HeartMuLa repository](https://github.com/HeartMuLa/heartlib) for licensing information.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via GitHub Issues
- Suggest new features
- Submit pull requests

## 📧 Support

For questions and support:
- Open an issue on GitHub
- Check the [HeartMuLa documentation](https://github.com/HeartMuLa/heartlib)

---

**Made with ❤️ using HeartMuLa | Developed with assistance from [Claude Code](https://claude.ai/code)**

*Create music with AI, own your creativity*
