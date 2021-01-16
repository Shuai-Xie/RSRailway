import numpy as np

# rail/train seg dataset
train_rail_stats = {
    'rail': {
        'anns': 3726,
        'imgs': 998,
    },
    'train': {
        'anns': 858,
        'imgs': 477,
    },  # 474 两个场景都有
}

train_rail_config = {
    'cnts': np.array([4.67970128e+08, 4.17537370e+07, 9.19453500e+06]),
    'freqs': np.array([0.90181834, 0.08046301, 0.01771865]),  # 9:1
    # 'weights': np.array([1.53075645, 10.44593591, 27.0089962]),
    'weights': np.array([3.53075645, 10.44593591, 17.0089962]),
    'label_names': ['bg', 'rail', 'train'],
    'label_colors': [
        (0, 0, 0),  # bg=0; 作为 0 类
        (0, 0, 255),  # rail=1
        (0, 255, 0),  # train=2
        # (255, 0, 0),  # obstacle=3
    ]
}
