import torch


def points_to_spatial_tree_codes(points, size, depth):
    """
    Turn a point into discrete codes that represent the path through a spatial tree.
    """
    points = points.clone().T
    dims, batch_size = points.shape
    code = torch.zeros((depth, batch_size), dtype=torch.long, device=points.device)
    indices = torch.zeros(batch_size, dtype=torch.long, device=points.device)

    for i in range(depth):
        size /= 2.0

        for j in range(dims):
            indices |= (points[j] >= 0) << j
            points[j] = (points[j] >= 0) * (points[j] - size) + (points[j] < 0) * (points[j] + size)

        code[i] = indices
        indices.fill_(0)

    return code.T


if __name__ == '__main__':
    points = torch.tensor([
        [1.0, -2.0, 3.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [-10.0, 0.0, 0.0],
        [0.0, -10.0, 0.0],
        [0.0, 0.0, -10.0],
        [-10.0, 10.0, 10.0],
        [10.0, -10.0, 10.0],
        [10.0, 10.0, -10.0],
    ])

    size = 10.0
    depth = 10

    print(points_to_spatial_tree_codes(points, size, depth))
