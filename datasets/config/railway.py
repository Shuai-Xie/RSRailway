category = [
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
    'train',  # 添加火车和轨道
    'rail',
]
color_pans = [
    (204, 78, 210),
    (0, 192, 255),
    (0, 131, 0),
    (240, 176, 0),
    (254, 100, 38),
    (0, 0, 255),
    (182, 117, 46),
    (185, 60, 129),
    (204, 153, 255),
    (80, 208, 146),
    (0, 0, 204),
    (17, 90, 197),
    (0, 255, 255),
    (102, 255, 102),
    (255, 255, 0),
    (255, 0, 0),  # 红
    (0, 255, 0),  # 绿
]

color_map = {k: v for k, v in zip(category, color_pans)}
