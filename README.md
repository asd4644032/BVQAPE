# BVQAPE

## Project Introduction
This project provides a variety of tools and scripts for image generation, processing, evaluation, and prompt generation, suitable for multimodal AI tasks.

---

## Directory Structure
- `tools/`: Various models and tool scripts
- `others/`: Scripts related to prompt generation and refinement
- `evaluate/`: Scripts and data for evaluation and statistics
- `images/`: Example images
- `utils/`: Helper tools and check scripts
- Main Python scripts (e.g., `generate_image.py`, `llm.py`, etc.)
- JSON, TXT, CSV, and other data files

---

## Installation

1.  **Install Anaconda or Miniconda:**

    * Download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Miniconda is a smaller distribution that includes only Conda and its dependencies.
2.  **Create a Conda environment:**

    ```bash
    conda create --name bvqape python=3.12
    ```
3.  **Activate the Conda environment:**

    ```bash
    conda activate bvqape
    ```
4.  **Install PyTorch:**

    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
    ```
    * Adjust the `cu126` based on your CUDA version, or use `cpu` if you don't have a NVIDIA GPU.
5.  **Clone the ComfyUI repository:**

    ```bash
    git clone [https://github.com/comfyanonymous/ComfyUI.git](https://github.com/comfyanonymous/ComfyUI.git)
    cd ComfyUI
    ```
6.  **Install ComfyUI dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
7.  **Clone custom nodes:**

    ```bash
    cd custom_nodes
    git clone [https://github.com/stavsap/comfyui-ollama.git](https://github.com/stavsap/comfyui-ollama.git)
    git clone [https://github.com/fairy-root/ComfyUI-Show-Text.git](https://github.com/fairy-root/ComfyUI-Show-Text.git)
    ```
8.  **Install custom node dependencies:**

    ```bash
    cd comfyui-ollama
    pip install -r requirements.txt
    cd ..
    cd ..
    ```

---

## Execution Example
For `generate_image.py`:
```bash
python generate_image.py --help
