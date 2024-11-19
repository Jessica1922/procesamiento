import plotly.graph_objects as go
import plotly.express as px



def create_output_directories(base_path):
    """
    Creates necessary output directories for data img and notebooks.
    """
    folders_to_create = ['data', 'img', 'notebooks']
    list_of_folders = []
    for main_folder in folders_to_create:
        folder_path = base_path / main_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        list_of_folders.append(folder_path)
    return list_of_folders

def create_data_directories(base_path):
    """
    Creates necessary output directories for data raw and processed.
    """
    folders_to_create = ['raw', 'processed']
    list_of_folders = []
    for main_folder in folders_to_create:
        folder_path = base_path / main_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        list_of_folders.append(folder_path)
    return list_of_folders

def create_img_directories(base_path):
        """
        Crea los directorios necesarios para img.
        """
        folder_path = base_path / 'img'
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

def plot_training_history(history,name):
    """
    Plot the training and validation loss history using Plotly.

    Parameters:
    - history : dict
        A dictionary containing the history of the training and validation losses.
        Typically, this is `history.history` where `history` is the output from a
        training process in libraries like TensorFlow/Keras.

    Returns:
    - A Plotly figure object that can be displayed with `fig.show()`.
    """

    train_loss = history['loss']
    val_loss = history['val_loss']

    train_trace = go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')
    )

    val_trace = go.Scatter(
        x=list(range(len(val_loss))),
        y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    )

    # Create layout
    layout = go.Layout(
        title=f'Training and Validation Loss with {name}',
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='Loss'),
        legend=dict(x=0.1, y=1.1, orientation='h')
    )

    # Create figure
    fig = go.Figure(data=[train_trace, val_trace], layout=layout)

    last_train_loss = train_loss[-1]
    last_val_loss   = val_loss[-1]
    fig.add_hline(y=last_train_loss, line_dash="dot", line_color="blue", annotation_text=f"Train Loss: {last_train_loss:.4f}",annotation_position="bottom right")
    fig.add_hline(y=last_val_loss, line_dash="dot", line_color="red", annotation_text=f"Validation Loss: {last_val_loss:.4f}",annotation_position="top right")
    # Adjust size and resolution
    fig.update_layout(width=640, height=480)  # equivalent to a 4x3 figure at 160 dpi
    fig.write_html(f"training_history_{name}.html")
    fig.show()

def plot_history_jan(histories):
    """
    Plot the training and validation loss history using Plotly.

    Parameters:
    - histories : dict
        A dictionary containing the histories of the training and validation losses.
        Each key is a name of the model and each value is a `History` object from
        TensorFlow/Keras or a dictionary equivalent.

    Returns:
    - None: Displays a Plotly figure for each history.
    """
    # Prepare colors
    top_n = len(histories)
    colors = px.colors.qualitative.Plotly[:top_n] if top_n <= len(px.colors.qualitative.Plotly) else px.colors.qualitative.Plotly * (top_n // len(px.colors.qualitative.Plotly) + 1)
    
    for i, (name, history) in enumerate(histories.items()):
        if isinstance(history, dict):
            train_loss = history['loss']
            val_loss = history.get('val_loss', [])
        else:  # Assuming history is a TensorFlow/Keras History object
            train_loss = history.history['loss']
            val_loss = history.history.get('val_loss', [])

        # Train loss trace
        train_trace = go.Scatter(
            x=list(range(len(train_loss))),
            y=train_loss,
            mode='lines',
            name='Train Loss',
            line=dict(color=colors[i * 2 % len(colors)])
        )

        # Validation loss trace
        val_trace = go.Scatter(
            x=list(range(len(val_loss))),
            y=val_loss,
            mode='lines',
            name='Validation Loss',
            line=dict(color=colors[(i * 2 + 1) % len(colors)])
        )

        # Create layout
        layout = go.Layout(
            title=f'Training and Validation Loss for {name}',
            xaxis=dict(title='Epochs'),
            yaxis=dict(title='Loss'),
            legend=dict(x=0.1, y=1.1, orientation='h')
        )

        # Create figure and add horizontal lines
        fig = go.Figure(data=[train_trace, val_trace], layout=layout)
        if train_loss:
            fig.add_hline(y=train_loss[-1], line_dash="dot", line_color=colors[i * 2 % len(colors)], annotation_text=f"Train Loss: {train_loss[-1]:.4f}", annotation_position="bottom right")
        if val_loss:
            fig.add_hline(y=val_loss[-1], line_dash="dot", line_color=colors[(i * 2 + 1) % len(colors)], annotation_text=f"Validation Loss: {val_loss[-1]:.4f}", annotation_position="top right")

        # Adjust size and resolution
        fig.update_layout(width=640, height=480)
        fig.write_html(f"training_history_{name}.html")
        fig.show()

def plot_history_combined(histories):
    """
    Plot the training and validation loss history on a single Plotly figure.

    Parameters:
    - histories : dict
        A dictionary containing the histories of the training and validation losses.
        Each key is a name of the model and each value is a `History` object from
        TensorFlow/Keras or a dictionary equivalent.

    Returns:
    - None: Displays a single Plotly figure with all histories.
    """
    # Prepare colors
    top_n = len(histories) * 2  # We need two colors per history (one for training, one for validation)
    colors = px.colors.qualitative.Plotly[:top_n] if top_n <= len(px.colors.qualitative.Plotly) else px.colors.qualitative.Plotly * (top_n // len(px.colors.qualitative.Plotly) + 1)
    
    # Create an empty figure
    fig = go.Figure()

    for i, (name, history) in enumerate(histories.items()):
        if isinstance(history, dict):
            train_loss = history['loss']
            val_loss = history.get('val_loss', [])
        else:  # Assuming history is a TensorFlow/Keras History object
            train_loss = history.history['loss']
            val_loss = history.history.get('val_loss', [])

        # Train loss trace
        train_trace = go.Scatter(
            x=list(range(len(train_loss))),
            y=train_loss,
            mode='lines',
            name=f'{name} Train Loss',
            line=dict(color=colors[i % len(colors)])
        )

        # Validation loss trace
        val_trace = go.Scatter(
            x=list(range(len(val_loss))),
            y=val_loss,
            mode='lines',
            name=f'{name} Validation Loss',
            line=dict(color=colors[i  % len(colors)]),
            line_dash="dot"
        )

        # last_train_loss = train_loss[-1]
        # last_val_loss   = val_loss[-1]
        # fig.add_hline(y=last_train_loss, line_dash="dot", line_color=colors[i % len(colors)], annotation_text=f"Train Loss: {last_train_loss:.4f}",annotation_position="bottom right")
        # fig.add_hline(y=last_val_loss, line_dash="dot", line_color=colors[i % len(colors)], annotation_text=f"Validation Loss: {last_val_loss:.4f}",annotation_position="top right")

        # Add traces to the figure
        fig.add_trace(train_trace)
        fig.add_trace(val_trace)

    # Update layout to accommodate all histories
    fig.update_layout(
        title='Training and Validation Loss for All Models',
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='Loss'),
        legend=dict(x=0.1, y=1.1, orientation='h')
    )

    # Show figure
    fig.show()
