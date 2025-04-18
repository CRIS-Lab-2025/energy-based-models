#!/usr/bin/env python
import numpy as np
import umap
import argparse
import base64
from PIL import Image
import io
import time
import os

# Conditional imports based on need
# TensorFlow only needed for data loading
from tensorflow.keras.datasets import mnist, fashion_mnist 

# Bokeh imports (only if needed)
try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

# Plotly imports (only if needed)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- Helper Functions ---

def np_image_to_base64(im_matrix):
    """Converts a NumPy array image to a base64 string for HTML embedding."""
    # Ensure correct mode ('L' for grayscale)
    im = Image.fromarray((im_matrix * 255).astype(np.uint8).squeeze(), mode='L') 
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    im_bytes = buffer.getvalue()
    im_base64 = base64.b64encode(im_bytes).decode('utf-8')
    return f'data:image/png;base64,{im_base64}'

def load_data(dataset_name):
    """Loads MNIST or Fashion-MNIST dataset."""
    print(f"Loading {dataset_name} data...")
    if dataset_name == 'mnist':
        (x_train, y_train), (_, _) = mnist.load_data()
        class_names = [str(i) for i in range(10)] # Labels 0-9
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (_, _) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'fashion_mnist'.")

    # Normalize and reshape (add channel dim if needed by models, not needed for raw viz)
    x_train = x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    
    print(f"Loaded {len(x_train)} training samples.")
    return x_train, y_train, class_names

def get_sampled_data(x_data, y_data, n_samples_per_class=500):
    """Samples data evenly across classes."""
    print(f"Sampling {n_samples_per_class} points per class...")
    unique_labels = np.unique(y_data)
    n_classes = len(unique_labels)
    
    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(y_data == label)[0]
        n_available = len(label_indices)
        n_to_sample = min(n_samples_per_class, n_available)
        
        if n_to_sample > 0:
            sampled_indices = np.random.choice(label_indices, n_to_sample, replace=False)
            selected_indices.extend(sampled_indices)
        else:
             print(f"Warning: No samples available for class {label} to sample from.")
            
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    print(f"Total samples selected: {len(selected_indices)}")
    # Return original indices as well
    original_indices = selected_indices 
    return x_data[selected_indices], y_data[selected_indices], original_indices 

# --- Bokeh 2D Visualization --- 
def create_umap_bokeh_2d(embedding, images, labels, indices, class_names, title):
    """Creates the 2D Bokeh plot with image hover."""
    if not BOKEH_AVAILABLE:
        print("Error: Bokeh library not found. Cannot create Bokeh plot.")
        return None
        
    print("Creating Bokeh plot...")
    print("Converting images to base64...")
    start_time = time.time()
    hover_images_base64 = [np_image_to_base64(img) for img in images]
    print(f"Base64 conversion finished in {time.time() - start_time:.2f} seconds.")

    color_palette = Category10[10]
    colors = [color_palette[label % 10] for label in labels] # Use modulo for safety

    source = ColumnDataSource(
        data=dict(
            x=embedding[:, 0], 
            y=embedding[:, 1],
            image=hover_images_base64, 
            label=[class_names[l % 10] for l in labels], # Use modulo for safety
            label_index=labels,
            index=indices,
            color=colors
        )
    )
    
    tooltips = """
        <img src="@image" height='50' width='50' style='image-rendering: pixelated;'><br>
        <b>Class:</b> @label (@label_index)<br>
        <b>Index:</b> @index
    """
    hover = HoverTool(tooltips=tooltips)
    
    p = figure(tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"], title=title)
    p.scatter(x='x', y='y', source=source, size=4, alpha=0.6,
              color='color', legend_field='label',
              hover_fill_color="firebrick", hover_alpha=1.0)
    p.xaxis.axis_label = "UMAP1"
    p.yaxis.axis_label = "UMAP2"
    p.legend.title = 'Class'
    p.legend.location = "top_left"
    p.legend.orientation = "vertical"
    p.legend.click_policy="hide"
    return p

# --- Plotly 3D Visualization ---
def create_umap_plotly_3d(embedding, labels, indices, class_names, title):
    """Creates the 3D Plotly plot with text hover."""
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly library not found. Cannot create Plotly plot.")
        return None
        
    print("Creating Plotly plot...")

    # Prepare customdata for hover: [label_name, label_index, original_index]
    custom_data = np.stack([
        [class_names[l % 10] for l in labels], 
        labels.astype(str), 
        indices.astype(str)
    ], axis=-1)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=3, # Smaller size often better for 3D
            color=labels,         
            colorscale='Viridis', 
            opacity=0.7,
            colorbar=dict(title='Class'), 
            showscale=True
        ),
        customdata=custom_data,
        hovertemplate=(
            "<b>Class: %{customdata[0]} (%{customdata[1]})</b><br>" +
            "Index: %{customdata[2]}<extra></extra>"
        ),
        hoverinfo='none' # Rely only on hovertemplate
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'),
        legend_title_text='Class',
        scene_camera_eye=dict(x=1.8, y=1.8, z=0.6) # Adjust camera angle
    )
    return fig

# --- Main Execution ---
def main(args):
    # Load data
    x_data, y_data, class_names = load_data(args.dataset)
    
    # Sample data
    sampled_images, sampled_labels, sampled_indices = get_sampled_data(
        x_data, y_data, n_samples_per_class=args.samples_per_class
    )
    
    # Prepare features for UMAP (raw pixels)
    print("\nUsing raw sampled images as features for UMAP.")
    features = sampled_images.reshape(len(sampled_images), -1)
    
    # --- Run UMAP --- 
    n_components = 3 if args.viz_type == 'plotly_3d' else 2
    print(f"Starting UMAP embedding ({n_components} components) for {len(features)} samples...")
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1, verbose=True)
    embedding = reducer.fit_transform(features)
    print(f"UMAP embedding finished in {time.time() - start_time:.2f} seconds.")

    # --- Generate Plot ---
    if args.viz_type == 'bokeh_2d':
        title = f"{args.dataset.capitalize()} UMAP (Bokeh 2D, {len(embedding)} Samples)"
        plot = create_umap_bokeh_2d(embedding, sampled_images, sampled_labels, sampled_indices, class_names, title)
        if plot:
            filename = f"{args.dataset}_bokeh_2d_{len(embedding)}.html"
            output_file(filename)
            save(plot, title=title)
            print(f"Bokeh 2D visualization saved as '{filename}'")
            
    elif args.viz_type == 'plotly_3d':
        # Filter data if specific classes are requested
        target_labels = sampled_labels
        target_embedding = embedding
        target_indices = sampled_indices
        class_subset_str = "all"
        if args.classes:
            print(f"Filtering for classes: {args.classes}")
            mask = np.isin(sampled_labels, args.classes)
            target_labels = sampled_labels[mask]
            target_embedding = embedding[mask]
            target_indices = sampled_indices[mask]
            print(f"Selected {len(target_labels)} samples for plotting.")
            class_subset_str = "_cls_" + "_".join(map(str, args.classes))
            
        title = f"{args.dataset.capitalize()} UMAP (Plotly 3D, {len(target_embedding)} Samples, Classes: {args.classes or 'All'})"
        plot = create_umap_plotly_3d(target_embedding, target_labels, target_indices, class_names, title)
        if plot:
            filename = f"{args.dataset}_plotly_3d{class_subset_str}_{len(target_embedding)}.html"
            plot.write_html(filename)
            print(f"Plotly 3D visualization saved as '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate UMAP visualizations for MNIST/Fashion-MNIST.')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('--viz_type', type=str, required=True, choices=['bokeh_2d', 'plotly_3d'], help='Type of visualization')
    parser.add_argument('--samples_per_class', type=int, default=500, help='Number of samples per class')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='List of specific classes to plot (only for plotly_3d, default: all sampled)')
    
    args = parser.parse_args()
    
    # Check library availability
    if args.viz_type == 'bokeh_2d' and not BOKEH_AVAILABLE:
        print("Error: Bokeh is required for bokeh_2d visualization but not installed.")
        exit(1)
    if args.viz_type == 'plotly_3d' and not PLOTLY_AVAILABLE:
        print("Error: Plotly is required for plotly_3d visualization but not installed.")
        exit(1)
        
    main(args) 