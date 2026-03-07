
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

def apply_interface_config(node, config: dict):
    """
    Map YAML config to Nipype node inputs automatically.
    
    Parameters
    ----------
    node : nipype.Node
        Node whose interface inputs should be set
    config : dict
        YAML section for this interface
    """

    iface = node.interface

    # Get valid input names from the interface
    valid_inputs = set(iface.inputs.traits().keys())
    print(f"Valid inputs for {node.name}: {valid_inputs}")

    params = {}

    steps = config.get("steps", {})
    args = config.get("args", {})

    for step, enabled in steps.items():

        # Map step flag to interface flag if present
        enable_key = f"enable_{step}"
        if enable_key in valid_inputs:
            params[enable_key] = enabled

        if not enabled:
            continue

        step_args = args.get(step, {})

        for k, v in step_args.items():
            print(f"Step '{step}' argument '{k}': {v}")
            if k in valid_inputs:
                params[k] = v
            else:
                print(f"Warning: '{k}' is not a valid input for {node.name} interface.")

    set_node_inputs(node, params)

    return node