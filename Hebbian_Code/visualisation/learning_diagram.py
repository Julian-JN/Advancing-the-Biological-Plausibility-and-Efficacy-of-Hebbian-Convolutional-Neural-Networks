from graphviz import Digraph


def create_hebbian_pipeline():
    # Initialize diagram with modified sizing
    dot = Digraph(comment='Hebbian Learning Pipeline')
    dot.attr(
        rankdir='LR',
        margin='0.05',
        bgcolor='#f9fafb',
        pad='0.5',
        splines='ortho',
        fontname='Arial',
        size='14,8',    # Further reduced width
        dpi='300'
    )

    # Define style attributes with narrower widths
    styles = {
        'io': {
            'fillcolor': '#ecfdf5',
            'color': '#86efac',
            'width': '1.5',    # Reduced width
            'height': '1.2',
            'penwidth': '1.5'
        },
        'processing': {
            'fillcolor': '#eef2ff',
            'color': '#a5b4fc',
            'width': '1.5',
            'height': '1.2',
            'penwidth': '1.5'
        },
        'learning': {
            'fillcolor': '#fff7ed',
            'color': '#fdba74',
            'width': '1.5',
            'height': '1.2',
            'penwidth': '1.5'
        }
    }

    # Common node attributes
    dot.attr('node',
             shape='rect',
             style='rounded,filled',
             fontname='Arial',
             fontsize='20',
             margin='0.3,0.2'
             )

    # Edge styling
    dot.attr('edge',
             color='#94a3b8',
             penwidth='1.5',
             arrowsize='0.8'
             )

    # Title
    with dot.subgraph(name='cluster_title') as title:
        title.attr(label='Hebbian Learning Pipeline in CNN Layer',
                  labelloc='t',
                  fontsize='22',
                  fontname='Arial Bold')

    # Input Section
    with dot.subgraph(name='cluster_input') as input_cluster:
        input_cluster.attr(label='INPUT',
                         labeljust='l',
                         fontcolor='#047857',
                         fontname='Arial Bold',
                         fontsize='20',
                         margin='15')
        input_cluster.node('input', 'Input\nData', **styles['io'])

    # Processing and Learning Steps
    dot.node('presynaptic',
             'Pre-\nsynaptic\nCompetition',
             **styles['processing'])

    dot.node('postsynaptic',
             'Post-\nsynaptic\nActivity',
             **styles['learning'])

    dot.node('lateral',
             'Lateral\nInhibition',
             **styles['processing'])

    # Competition Mechanisms Box
    dot.node('competition',
             'Competition\nMethods\n\nHard-WTA\nSoft-WTA\nTemporal\nHomeostatic',
             shape='box',
             style='rounded,filled',
             fillcolor='#fff7ed',
             color='#fdba74',
             fontname='Arial',
             fontsize='20',
             margin='0.3',
             height='2.5',
             width='2.0')  # Reduced width

    dot.node('weight_update',
             'Hebbian\nWeight\nUpdate',
             **styles['learning'])

    dot.node('normalization',
             'Weight\nNormal-\nization',
             **styles['processing'])

    # Output Section
    with dot.subgraph(name='cluster_output') as output_cluster:
        output_cluster.attr(label='OUTPUT',
                          labeljust='l',
                          fontcolor='#047857',
                          fontname='Arial Bold',
                          fontsize='20',
                          margin='15')
        output_cluster.node('output', 'Feature\nMaps', **styles['io'])

    # Create edges
    edges = [
        ('input', 'presynaptic'),
        ('presynaptic', 'postsynaptic'),
        ('postsynaptic', 'lateral'),
        ('lateral', 'competition'),
        ('competition', 'weight_update'),
        ('weight_update', 'normalization'),
        ('normalization', 'output')
    ]

    for edge in edges:
        dot.edge(*edge, minlen='1')

    return dot


def save_pipeline(filename='hebbian_pipeline_vertical'):
    """
    Creates and saves the Hebbian pipeline diagram with vertical text.

    Args:
        filename (str): Base name for the output files (without extension)
    """
    pipeline = create_hebbian_pipeline()

    # Save in multiple formats
    formats = ['pdf', 'png']
    for fmt in formats:
        pipeline.render(filename,
                       view=(fmt == 'pdf'),
                       format=fmt,
                       cleanup=True)

    print(f"Pipeline diagram saved as {filename}.pdf and {filename}.png")


if __name__ == '__main__':
    save_pipeline()