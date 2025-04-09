import umap
import h5py
import numpy as np
# import matplotlib.pyplot as plt  # Remove or comment out matplotlib
# from mpl_toolkits.mplot3d import Axes3D # Remove or comment out matplotlib 3D
import plotly.graph_objects as go # Import plotly
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script directory
forward_h5_path = os.path.join(script_dir, 'forward_data.h5')
backward_h5_path = os.path.join(script_dir, 'backward_data.h5')
# output_png_path = os.path.join(script_dir, 'umap_basin_3d.png') # No longer saving PNG
output_html_path = os.path.join(script_dir, 'umap_basin_3d.html') # Save as HTML

# Load data from HDF5 files
with h5py.File(forward_h5_path, 'r') as f_forward, h5py.File(backward_h5_path, 'r') as f_backward:
    # Load data in chunks to save memory
    chunk_size = 1000
    
    # Get total sizes
    forward_size = f_forward['hidden_states'].shape[0]
    backward_size = f_backward['hidden_states'].shape[0]
    
    # Initialize UMAP model for 3 components
    umap_model = umap.UMAP(n_components=3, random_state=42)
    
    # Process forward data in chunks for fitting UMAP
    all_forward_data = []
    all_forward_labels = []
    
    for i in range(0, forward_size, chunk_size):
        end_idx = min(i + chunk_size, forward_size)
        # Load and flatten chunk
        chunk_data = f_forward['hidden_states'][i:end_idx]
        chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
        all_forward_data.append(chunk_data_flat)
        
        # Load labels
        chunk_labels = f_forward['labels'][i:end_idx].argmax(axis=1)
        all_forward_labels.append(chunk_labels)
    
    # Concatenate all chunks
    forward_data_flat = np.vstack(all_forward_data)
    forward_labels = np.concatenate(all_forward_labels)
    
    # Fit UMAP on forward data
    umap_result = umap_model.fit_transform(forward_data_flat)
    
    # Process backward data in chunks for transforming with UMAP
    all_backward_data = []
    all_backward_labels = []
    
    for i in range(0, backward_size, chunk_size):
        end_idx = min(i + chunk_size, backward_size)
        # Load and flatten chunk
        chunk_data = f_backward['hidden_states'][i:end_idx]
        chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
        all_backward_data.append(chunk_data_flat)
        
        # Load labels
        chunk_labels = f_backward['labels'][i:end_idx].argmax(axis=1)
        all_backward_labels.append(chunk_labels)
    
    # Concatenate all chunks
    backward_data_flat = np.vstack(all_backward_data)
    backward_labels = np.concatenate(all_backward_labels)
    
    # Transform backward data using the fitted UMAP model
    umap_result_backward = umap_model.transform(backward_data_flat)

    # --- Plotly 3D Scatter Plot ---
    fig = go.Figure()

    # Add Forward (Real) data trace
    fig.add_trace(go.Scatter3d(
        x=umap_result[:, 0], 
        y=umap_result[:, 1], 
        z=umap_result[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=forward_labels,                # set color to an array/list of desired values
            colorscale='Viridis',             # choose a colorscale
            opacity=0.6,
            colorbar=dict(title='Class')      # Add colorbar title
        ),
        name='Forward (Real)'                   # Name for legend
    ))

    # Add Backward (Generated) data trace
    fig.add_trace(go.Scatter3d(
        x=umap_result_backward[:, 0], 
        y=umap_result_backward[:, 1], 
        z=umap_result_backward[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=backward_labels,
            colorscale='Viridis', 
            opacity=0.6,
            symbol='diamond'                   # Use different symbol
        ),
        name='Backward (Generated)'
    ))

    # Update layout
    fig.update_layout(
        title='Interactive 3D UMAP Projection with Class Labels',
        scene=dict(
            xaxis_title='UMAP Dim 1',
            yaxis_title='UMAP Dim 2',
            zaxis_title='UMAP Dim 3'
        ),
        margin=dict(r=20, b=10, l=10, t=40) # Adjust margins
    )

    # Save as interactive HTML
    fig.write_html(output_html_path)
    print(f"Interactive plot saved to {output_html_path}")

    # Remove plt.show() if you don't want matplotlib window
    # plt.show() 
