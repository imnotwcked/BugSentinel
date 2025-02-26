def main(args):
    global header_dir, param_dir
    header_dir = args.header_dir
    param_dir = args.param_dir

    # Create a new graph for the modified simple DNN
    graph = int_test.Graph()

    # Append weights from trained model to the graph
    graph.append('input', weight)
    graph.append('input_reshape', int_test.input_reshape, 'input')

    # Modify to add layers for the simple DNN (e.g., conv1, fc1, etc.)
    graph.append('fc1_w', weight)
    graph.append('fc1_b', weight)

    graph.append('fc2_w', weight)
    graph.append('fc2_b', weight)

    # Write headers for the fully connected layers
    write_header('fc1', [
        ('fc1_w', graph.eval('fc1_w'), 'FC', False), 
        ('fc1_b', graph.eval('fc1_b'), 'FC', False)])

    write_header('fc2', [
        ('fc2_w', graph.eval('fc2_w'), 'FC', False), 
        ('fc2_b', graph.eval('fc2_b'), 'FC', False)])

    print("Headers written for the modified DNN structure.")
