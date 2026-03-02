
def set_node_inputs(node, inputs_dict, overwrite=True):
    """
    Helper function to set multiple inputs on a Nipype node from a dictionary.

    Args:
        node: The Nipype node whose inputs you want to set.
        inputs_dict: Dictionary of {input_name: value} pairs.
        overwrite: If False, only set inputs that are not already set.
    """
    for key, value in inputs_dict.items():
        if overwrite or not hasattr(node.inputs, key):
            if hasattr(node.inputs, key):
                setattr(node.inputs, key, value)
            else:
                print(f"Warning: '{key}' is not a valid input for this node.")