def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments from a dot-separated path string.
    Positive values shave the first segments, negative shave the last segments.

    Args:
        path (str): The dot-separated path string.
        n_shave_prefix_segments (int): Number of segments to remove.
                                       If positive, removes from the beginning.
                                       If negative, removes from the end.

    Returns:
        str: The modified path string.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attention layers to the new naming scheme (local renaming).

    Args:
        old_list (list): List of old paths.
        n_shave_prefix_segments (int): Number of prefix segments to shave.

    Returns:
        list: A list of dictionaries mapping old paths to new paths.
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnet layers to the new naming scheme (local renaming).

    Args:
        old_list (list): List of old paths.
        n_shave_prefix_segments (int): Number of prefix segments to shave.

    Returns:
        list: A list of dictionaries mapping old paths to new paths.
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping
