# Energy-Based Models Project

This repository contains code for exploring Energy-Based Models (EBMs), particularly in the context of Equilibrium Propagation.

## Generating UMAP Visualizations

The Bokeh 2D UMAP visualizations show embeddings of sampled MNIST and Fashion-MNIST data, colored by class label, with image previews on hover.

**Prerequisites:**

1.  Ensure you have a Python environment set up (e.g., the `ebm_venv` virtual environment in the parent directory).
2.  Install the necessary dependencies:
    ```bash
    pip install umap-learn tensorflow bokeh Pillow
    ```
    *Note: You might need to install these within your specific virtual environment.*

**Generating the Plots:**

The visualizations are generated using the `visualize_umap.py` script located in the parent directory (`../..` relative to this README).

1.  **Navigate** to the directory containing `visualize_umap.py` (i.e., the workspace root `/home/kheri/dev/CRIS/`).
2.  **Run the script** using the Python interpreter from your environment. Specify the dataset and visualization type.

    *   **For MNIST:**
        ```bash
        # Example assuming ebm_venv is in the parent directory
        ../ebm_venv/bin/python visualize_umap.py --dataset mnist --viz_type bokeh_2d --samples_per_class 500
        ```
        *(If your environment activation works correctly, you might be able to use `python visualize_umap.py ...` after activating)*

    *   **For Fashion-MNIST:**
        ```bash
        # Example assuming ebm_venv is in the parent directory
        ../ebm_venv/bin/python visualize_umap.py --dataset fashion_mnist --viz_type bokeh_2d --samples_per_class 500
        ```

3.  **Output:** The script will save the generated HTML files in the directory where it was run (e.g., `mnist_bokeh_2d_5000.html` and `fashion_mnist_bokeh_2d_5000.html` in the workspace root).

*(Optional: You can copy these generated files into the `visualizations/` directory within this repository if desired.)*

## Other Project Components

*(Add more details about the EBM code, experiments, etc. here)* 