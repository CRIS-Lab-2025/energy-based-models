# Energy-Based Models Project

This repository contains code for exploring Energy-Based Models (EBMs), particularly in the context of Equilibrium Propagation.

## Generating UMAP Visualizations

The Bokeh 2D UMAP visualizations show embeddings of sampled MNIST and Fashion-MNIST data, colored by class label, with image previews on hover.

**Prerequisites:**

1.  Ensure you have a Python environment set up. This project often uses a virtual environment named `ebm_venv` located in the parent directory (`../ebm_venv/`).
2.  **Activate** your chosen Python environment (e.g., `source ../ebm_venv/bin/activate`).
3.  Install the necessary dependencies within the activated environment:
    ```bash
    pip install umap-learn tensorflow bokeh Pillow
    ```

**Generating the Plots:**

The visualizations are generated using the `visualize_umap.py` script located in this repository's root directory.

1.  Ensure your environment is activated (see Prerequisites).
2.  **Run the script** from this repository's root directory. Specify the dataset and visualization type.

    *   **For MNIST:**
        ```bash
        python visualize_umap.py --dataset mnist --viz_type bokeh_2d --samples_per_class 500
        ```

    *   **For Fashion-MNIST:**
        ```bash
        python visualize_umap.py --dataset fashion_mnist --viz_type bokeh_2d --samples_per_class 500
        ```

3.  **Output:** The script will save the generated HTML files in this repository's root directory (e.g., `mnist_bokeh_2d_5000.html`).

*(Optional: You can move these generated files into the `visualizations/` directory within this repository if desired.)*

## Other Project Components

*(Add more details about the EBM code, experiments, etc. here)* 