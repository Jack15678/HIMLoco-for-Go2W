"""Exact square stair flights around the existing 3 m central platform."""
import numpy as np


def stair_height(xy, size, height, depth, platform=3.):
    count = int(np.ceil((size-platform)/2/depth))
    radius = np.abs(np.asarray(xy)-size/2).max(axis=-1)
    level = np.clip(count-1-np.floor((radius-platform/2)/depth+1e-7), 0, count)
    return level * height


def stair_mesh(size, height, depth, platform=3.):
    if not (size > platform > 0 and 0 < depth <= (size-platform)/2 and height != 0):
        raise ValueError('Invalid staircase dimensions')
    count = int(np.ceil((size-platform)/2/depth))
    vertices, triangles = [], []

    def quad(points):
        start = len(vertices)
        vertices.extend(points)
        triangles.extend([[start, start+1, start+2], [start, start+2, start+3]])

    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    center = corners * platform/2 + size/2
    quad(np.column_stack((center, np.full(4, count*height))))
    for step in range(count):
        inner = corners*(platform/2+step*depth)+size/2
        outer = corners*min(platform/2+(step+1)*depth, size/2)+size/2
        z = (count-step-1)*height
        for side in range(4):
            following = (side+1) % 4
            quad([[*inner[side], z], [*outer[side], z],
                  [*outer[following], z], [*inner[following], z]])
            quad([[*inner[side], z], [*inner[following], z],
                  [*inner[following], z+height], [*inner[side], z+height]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)
