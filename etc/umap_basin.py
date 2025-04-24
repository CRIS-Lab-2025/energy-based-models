import umap
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import plotly.express as px

def create_umap_visualization(label_filter=None):
    """
    Create UMAP visualization for data points, optionally filtering by a specific label.
    UMAP is only computed on the filtered data when a label is specified.
    
    Args:
        label_filter (int, optional): If provided, only points with this label will be processed and plotted.
    """
    # Load data from HDF5 files
    with h5py.File('forward_data.h5', 'r') as f_forward, h5py.File('backward_data.h5', 'r') as f_backward:
        # Load data in chunks to save memory
        chunk_size = 1000
        
        # Get total sizes
        forward_size = f_forward['hidden_states'].shape[0]
        backward_size = f_backward['hidden_states'].shape[0]
        
        # Process forward data in chunks and filter by label
        all_forward_data = []
        all_forward_labels = []
        
        for i in range(0, forward_size, chunk_size):
            end_idx = min(i + chunk_size, forward_size)
            # Load and flatten chunk
            chunk_data = f_forward['hidden_states'][i:end_idx]
            chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
            
            # Load labels
            chunk_labels = f_forward['labels'][i:end_idx].argmax(axis=1)
            
            # Filter by label if specified
            if label_filter is not None:
                mask = chunk_labels == label_filter
                chunk_data_flat = chunk_data_flat[mask]
                chunk_labels = chunk_labels[mask]
            
            if len(chunk_data_flat) > 0:
                all_forward_data.append(chunk_data_flat)
                all_forward_labels.append(chunk_labels)
        
        # Check if we have data
        if not all_forward_data:
            print(f"No forward data found for label {label_filter}")
            return
            
        # Concatenate all chunks
        forward_data_flat = np.vstack(all_forward_data)
        forward_labels = np.concatenate(all_forward_labels)
        
        # Process backward data in chunks and filter by label
        all_backward_data = []
        all_backward_labels = []
        
        for i in range(0, backward_size, chunk_size):
            end_idx = min(i + chunk_size, backward_size)
            # Load and flatten chunk
            chunk_data = f_backward['hidden_states'][i:end_idx]
            chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
            
            # Load labels
            chunk_labels = f_backward['labels'][i:end_idx].argmax(axis=1)
            
            # Filter by label if specified
            if label_filter is not None:
                mask = chunk_labels == label_filter
                chunk_data_flat = chunk_data_flat[mask]
                chunk_labels = chunk_labels[mask]
            
            if len(chunk_data_flat) > 0:
                all_backward_data.append(chunk_data_flat)
                all_backward_labels.append(chunk_labels)
        
        if not all_backward_data:
            print(f"No backward data found for label {label_filter}")
            return
            
        # Concatenate all chunks
        backward_data_flat = np.vstack(all_backward_data)
        backward_labels = np.concatenate(all_backward_labels)
        
        # Combine forward and backward data for UMAP fitting
        combined_data = np.vstack([forward_data_flat, backward_data_flat])
        
        # Initialize UMAP model and fit on the combined data
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_result_combined = umap_model.fit_transform(combined_data)
        
        # Split the results back to forward and backward
        n_forward = forward_data_flat.shape[0]
        umap_result_forward = umap_result_combined[:n_forward]
        umap_result_backward = umap_result_combined[n_forward:]
        
        # Prepare data for forward points
        forward_df = pd.DataFrame({
            'UMAP1': umap_result_forward[:, 0],
            'UMAP2': umap_result_forward[:, 1],
            'Label': forward_labels,
            'Type': 'Forward'
        })

        # Prepare data for backward points
        backward_df = pd.DataFrame({
            'UMAP1': umap_result_backward[:, 0],
            'UMAP2': umap_result_backward[:, 1],
            'Label': backward_labels,
            'Type': 'Backward'
        })

        # Combine forward and backward data
        combined_df = pd.concat([forward_df, backward_df], ignore_index=True)

        # Create title and filename based on filter
        label_name = f'label_{label_filter}' if label_filter is not None else 'all_labels'
        title = f'UMAP Projection for Label {label_filter}' if label_filter is not None else 'UMAP Projection with All Labels'
        filename = f'umap_basin_{label_name}.html'
        
        # Create scatter plot using Plotly
        fig = px.scatter(
            combined_df,
            x='UMAP1',
            y='UMAP2',
            color='Label',
            symbol='Type',
            title=title,
            labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
            color_continuous_scale='Viridis'
        )

        # Save the plot as an HTML file
        fig.write_html(filename)
        print(f"Plot saved as {filename}")

        # Show the plot
        fig.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate UMAP visualization of hidden states.')
    parser.add_argument('-l','--label', type=int, help='Specific label (number) to plot. If not provided, all labels will be plotted.', 
                        default=None)
    args = parser.parse_args()
    
    # Create visualization based on user input
    create_umap_visualization(label_filter=args.label)
