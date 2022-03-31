import numpy as np


def subsequence_indices_from(start_index: int, need_indices: int, seq_len: int, skip_prob: float):
    if start_index + need_indices > seq_len:
        raise RuntimeError(
            f"Incorrect indices. start_index={start_index}, seq_len={seq_len}, but need={need_indices}"
        )

    if np.isclose(skip_prob, 0):
        return list(range(start_index, start_index + need_indices))

    indices = [start_index]

    while len(indices) != need_indices:
        start_index += 1

        need = need_indices - len(indices)
        dist = seq_len - start_index

        random_value = np.random.rand() + 1e-18

        if dist == need or random_value > skip_prob:
            indices.append(start_index)
        elif dist < need:
            raise RuntimeError(
                f"Incorrect indices. start_index={start_index}, indices={indices}, seq_len={seq_len}"
            )

    return indices
