import numpy as np

# class ID → RGB color
CLASS_COLORS = [
    (128,  64, 128),  # 0 road
    (244,  35, 232),  # 1 sidewalk
    ( 70,  70,  70),  # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170,  30),  # 6 traffic light
    (220, 220,   0),  # 7 traffic sign
    (107, 142,  35),  # 8 vegetation
    (152, 251, 152),  # 9 terrain
    ( 70, 130, 180),  # 10 sky
    (220,  20,  60),  # 11 person
    (255,   0,   0),  # 12 rider
    (  0,   0, 142),  # 13 car
    (  0,   0,  70),  # 14 truck
    (  0,  60, 100),  # 15 bus
    (  0,  80, 100),  # 16 train
    (  0,   0, 230),  # 17 motorcycle
    (119,  11,  32),  # 18 bicycle
    (  0,   0,   0),  # 19 unknown
]

def color_mask_to_label(mask_np):
    h, w, _ = mask_np.shape
    lbl = np.zeros((h, w), dtype=np.int64)
    for idx, color in enumerate(CLASS_COLORS):
        matches = (
            (mask_np[:,:,0] == color[0]) &
            (mask_np[:,:,1] == color[1]) &
            (mask_np[:,:,2] == color[2])
        )
        lbl[matches] = idx
    return lbl

def label_to_color_mask(label_np):
    h, w = label_np.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        mask[label_np == idx] = color
    return mask
