    # Save each layer's weights and biases to separate pickle files
    layer_names = ['h1', 'h2', 'out']
    bias_names = ['b1', 'b2', 'out']
    for layer in layer_names:
        # Save weights
        filename_weight = "{}_weight.param".format(layer)
        with open(filename_weight, "wb") as f:
            pickle.dump(sess.run(weights[layer]), f)

    for bias in bias_names:
        # Save biases
        filename_bias = "{}_bias.param".format(bias)
        with open(filename_bias, "wb") as f:
            pickle.dump(sess.run(biases[bias]), f)