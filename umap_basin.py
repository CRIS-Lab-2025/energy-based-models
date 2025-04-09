import umap
import h5py
import numpy as np
# import matplotlib.pyplot as plt  # Remove or comment out matplotlib
# from mpl_toolkits.mplot3d import Axes3D # Remove or comment out matplotlib 3D
import plotly.graph_objects as go # Import plotly
import os
import argparse # Import argparse

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Generate UMAP visualizations for EBM hidden states.')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'single', 'forward', 'backward', 'certainty', 'confusable'],
                        help='Type of visualization to generate.')
    parser.add_argument('--digits', type=int, nargs='+', default=None,
                        help='Digit(s) to focus on for single, certainty, or confusable modes.')
    parser.add_argument('--dim', type=int, default=3, choices=[2, 3],
                        help='Number of UMAP dimensions (2 or 3).')
    parser.add_argument('--output', type=str, default='umap_plot',
                        help='Base name for the output HTML file.')
    parser.add_argument('--max_points', type=int, default=None,
                        help='Maximum number of points per dataset (forward/backward) to plot (for performance).')
    args = parser.parse_args()

    # Validate arguments
    if args.mode in ['single', 'certainty'] and (args.digits is None or len(args.digits) != 1):
        parser.error(f'Mode \'{args.mode}\' requires exactly one digit specified with --digits.')
    if args.mode == 'confusable' and (args.digits is None or len(args.digits) < 2):
        parser.error(f'Mode \'{args.mode}\' requires at least two digits specified with --digits.')
        
    return args

# --- Main Script Logic ---
if __name__ == "__main__":
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forward_h5_path = os.path.join(script_dir, 'forward_data.h5')
    backward_h5_path = os.path.join(script_dir, 'backward_data.h5')
    output_html_path = os.path.join(script_dir, f'{args.output}_{args.mode}{"_" + "".join(map(str, args.digits)) if args.digits else ""}_{args.dim}d.html')

    print(f"Running UMAP in mode: {args.mode}, Digits: {args.digits}, Dim: {args.dim}")
    print(f"Output will be saved to: {output_html_path}")

    # --- Data Loading and Filtering ---
    forward_states_list = []
    forward_labels_list = []
    backward_states_list = []
    backward_labels_list = [] # For argmax labels
    backward_probs_list = []  # For certainty mode
    backward_raw_labels_list = [] # For certainty mode raw targets

    chunk_size = 2000 # Process in chunks

    # Load Forward Data
    if args.mode in ['all', 'single', 'forward', 'confusable']:
        print("Loading forward data...")
        with h5py.File(forward_h5_path, 'r') as f_forward:
            forward_size = f_forward['hidden_states'].shape[0]
            for i in range(0, forward_size, chunk_size):
                end_idx = min(i + chunk_size, forward_size)
                labels_one_hot = f_forward['labels'][i:end_idx]
                states = f_forward['hidden_states'][i:end_idx]
                labels = np.argmax(labels_one_hot, axis=1)

                if args.mode == 'single' or args.mode == 'confusable':
                    mask = np.isin(labels, args.digits)
                    forward_states_list.append(states[mask])
                    forward_labels_list.append(labels[mask])
                else: # 'all' or 'forward'
                    forward_states_list.append(states)
                    forward_labels_list.append(labels)
        if not forward_states_list:
             print("Warning: No forward data matched the specified digits.")
        forward_states_all = np.vstack(forward_states_list) if forward_states_list else np.empty((0, 1000)) # Assuming 1000 hidden units
        forward_labels_all = np.concatenate(forward_labels_list) if forward_labels_list else np.empty((0,))
        print(f"Loaded {forward_states_all.shape[0]} forward samples.")

    # Load Backward Data
    if args.mode in ['all', 'single', 'backward', 'certainty', 'confusable']:
        print("Loading backward data...")
        with h5py.File(backward_h5_path, 'r') as f_backward:
            backward_size = f_backward['hidden_states'].shape[0]
            for i in range(0, backward_size, chunk_size):
                end_idx = min(i + chunk_size, backward_size)
                target_vectors = f_backward['labels'][i:end_idx]
                states = f_backward['hidden_states'][i:end_idx]
                labels = np.argmax(target_vectors, axis=1)

                if args.mode == 'single' or args.mode == 'confusable':
                    mask = np.isin(labels, args.digits)
                    backward_states_list.append(states[mask])
                    backward_labels_list.append(labels[mask])
                elif args.mode == 'certainty':
                    mask = (labels == args.digits[0])
                    backward_states_list.append(states[mask])
                    backward_labels_list.append(labels[mask]) # Keep argmax label for potential checks
                    backward_probs_list.append(target_vectors[mask, args.digits[0]])
                    backward_raw_labels_list.append(target_vectors[mask])
                else: # 'all' or 'backward'
                    backward_states_list.append(states)
                    backward_labels_list.append(labels)
        if not backward_states_list:
            print("Warning: No backward data matched the specified digits.")
        backward_states_all = np.vstack(backward_states_list) if backward_states_list else np.empty((0, 1000))
        backward_labels_all = np.concatenate(backward_labels_list) if backward_labels_list else np.empty((0,))
        backward_probs_all = np.concatenate(backward_probs_list) if backward_probs_list else np.empty((0,))
        backward_raw_labels_all = np.vstack(backward_raw_labels_list) if backward_raw_labels_list else np.empty((0,10))
        print(f"Loaded {backward_states_all.shape[0]} backward samples.")

    # Subsample if requested (to improve performance for large datasets)
    if args.max_points is not None:
        if forward_states_all.shape[0] > args.max_points:
            print(f"Subsampling forward data to {args.max_points} points.")
            indices = np.random.choice(forward_states_all.shape[0], args.max_points, replace=False)
            forward_states_all = forward_states_all[indices]
            forward_labels_all = forward_labels_all[indices]
        if backward_states_all.shape[0] > args.max_points:
            print(f"Subsampling backward data to {args.max_points} points.")
            indices = np.random.choice(backward_states_all.shape[0], args.max_points, replace=False)
            backward_states_all = backward_states_all[indices]
            backward_labels_all = backward_labels_all[indices]
            if args.mode == 'certainty':
                backward_probs_all = backward_probs_all[indices]
                backward_raw_labels_all = backward_raw_labels_all[indices]

    # --- UMAP Processing ---
    print("Running UMAP...")
    umap_model = umap.UMAP(n_components=args.dim, random_state=42, n_neighbors=15, min_dist=0.1, metric='euclidean')

    umap_forward = None
    umap_backward = None
    fit_data = []
    fit_labels = [] # For potential context, though not used for fitting
    
    # Decide what data to fit UMAP on
    if args.mode == 'forward':
        fit_data = forward_states_all
        fit_labels = forward_labels_all
    elif args.mode == 'backward' or args.mode == 'certainty':
        fit_data = backward_states_all
        fit_labels = backward_labels_all # or backward_probs_all for certainty
    else: # 'all', 'single', 'confusable' - fit primarily on forward data if available
        if forward_states_all.shape[0] > 0:
            fit_data = forward_states_all
            fit_labels = forward_labels_all
        elif backward_states_all.shape[0] > 0:
             fit_data = backward_states_all # Fallback if no forward data
             fit_labels = backward_labels_all
        else:
            print("Error: No data available to fit UMAP.")
            exit()

    if fit_data.shape[0] < umap_model.n_neighbors:
        print(f"Warning: Number of data points ({fit_data.shape[0]}) is less than n_neighbors ({umap_model.n_neighbors}). Reducing n_neighbors.")
        umap_model.n_neighbors = fit_data.shape[0] - 1
        
    if fit_data.shape[0] <= 1:
        print("Error: Not enough data points to run UMAP.")
        exit()

    print(f"Fitting UMAP on {fit_data.shape[0]} points...")
    umap_fitted = umap_model.fit_transform(fit_data)

    # Assign results based on fit data
    if args.mode == 'forward':
        umap_forward = umap_fitted
    elif args.mode == 'backward' or args.mode == 'certainty':
        umap_backward = umap_fitted
    else:
        if forward_states_all.shape[0] > 0:
            umap_forward = umap_fitted
            # Transform backward data if it exists
            if backward_states_all.shape[0] > 0:
                print("Transforming backward data...")
                umap_backward = umap_model.transform(backward_states_all)
        elif backward_states_all.shape[0] > 0: # Only backward data was available for fit
            umap_backward = umap_fitted

    # --- Plotting ---
    print("Generating plot...")
    fig = go.Figure()
    plot_title = f'{args.dim}D UMAP Projection ({args.mode})' 
    if args.digits:
         plot_title += f' - Digits: {args.digits}'
         
    color_axis_label = 'Class'
    colorscale = 'Viridis'

    # Add Forward Trace
    if umap_forward is not None and umap_forward.shape[0] > 0:
        marker_color = forward_labels_all
        fig.add_trace(go.Scatter3d(
            x=umap_forward[:, 0], 
            y=umap_forward[:, 1], 
            z=umap_forward[:, 2] if args.dim == 3 else None,
            mode='markers',
            marker=dict(
                size=3,
                color=marker_color,
                colorscale=colorscale, 
                opacity=0.7,
                colorbar=dict(title=color_axis_label) if args.mode != 'certainty' else None # Colorbar only if not certainty mode yet
            ),
            name='Forward (Real)',
            customdata=forward_labels_all, # Store labels for potential hover
            hovertemplate='<b>Digit:</b> %{customdata}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<br>Dim3: %{z:.2f}<extra></extra>' if args.dim == 3 else '<b>Digit:</b> %{customdata}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>'
        ))

    # Add Backward Trace
    if umap_backward is not None and umap_backward.shape[0] > 0:
        marker_symbol = 'diamond'
        marker_color = backward_labels_all
        hover_labels = backward_labels_all
        hover_template = '<b>Digit:</b> %{customdata}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<br>Dim3: %{z:.2f}<extra></extra>' if args.dim == 3 else '<b>Digit:</b> %{customdata}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>'

        if args.mode == 'certainty':
            marker_color = backward_probs_all
            color_axis_label = 'Target Probability'
            colorscale = 'Plasma' # Use a different scale for probability
            hover_labels = backward_probs_all
            hover_template = '<b>Prob:</b> %{customdata:.2f}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<br>Dim3: %{z:.2f}<extra></extra>' if args.dim == 3 else '<b>Prob:</b> %{customdata:.2f}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>'

        fig.add_trace(go.Scatter3d(
            x=umap_backward[:, 0], 
            y=umap_backward[:, 1], 
            z=umap_backward[:, 2] if args.dim == 3 else None,
            mode='markers',
            marker=dict(
                size=3,
                color=marker_color,
                colorscale=colorscale, 
                opacity=0.7,
                symbol=marker_symbol,
                colorbar=dict(title=color_axis_label) # Add colorbar for backward or certainty
            ),
            name='Backward (Generated)',
            customdata=hover_labels,
            hovertemplate=hover_template
        ))

    # Update layout
    layout_args = dict(
        title=plot_title,
        margin=dict(r=20, b=10, l=10, t=50) # Adjust margins
    )
    if args.dim == 3:
        layout_args['scene'] = dict(
                xaxis_title='UMAP Dim 1',
                yaxis_title='UMAP Dim 2',
                zaxis_title='UMAP Dim 3'
            )
    else: # 2D layout
         layout_args['xaxis_title'] = 'UMAP Dim 1'
         layout_args['yaxis_title'] = 'UMAP Dim 2'
         layout_args['width'] = 800
         layout_args['height'] = 700

    fig.update_layout(**layout_args)
    
    # Save as interactive HTML
    fig.write_html(output_html_path)
    print(f"Interactive plot saved to {output_html_path}")

    # Remove plt.show() if you don't want matplotlib window
    # plt.show() 
