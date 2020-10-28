# detect
dec_label_names = [
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
dec_label_colors = [
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

dec_color_map = {k: v for k, v in zip(dec_label_names, dec_label_colors)}

# segment
seg_label_names = ['bg', 'rail', 'plant', 'buildings', 'road', 'land', 'water', 'train']
seg_label_names_zh = ['背景', '铁轨', '绿植', '房屋', '道路', '黄土地', '水域', '火车']

seg_map_zh = {
    cat: cat_zh for cat, cat_zh in zip(seg_label_names, seg_label_names_zh)
}

seg_label_colors = [
    (0, 0, 0),
    (0, 0, 255),  # 铁轨
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),  # road 公路   粉
    (255, 255, 0),  # land 黄土地  黄
    (0, 255, 255),  # water
    (128, 128, 128),  # train
]

seg_color_map = {k: v for k, v in zip(seg_label_names, seg_label_names)}

# classify

cls_label_names = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
                   'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
                   'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
                   'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential',
                   'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland', 'dust']
