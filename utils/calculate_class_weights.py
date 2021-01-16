import numpy as np
from tqdm import tqdm
from pprint import pprint


def calculate_class_weights(dataloader, num_classes, save_path=None):
    z = np.zeros((num_classes,))
    tqdm_batch = tqdm(dataloader)
    tqdm_batch.set_description('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['target']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()

    # freqs
    total_frequency = np.sum(z)
    freqs = z / total_frequency

    # weights
    class_weights = []
    for freq in freqs:
        w = 1 / (np.log(1.02 + freq))
        class_weights.append(w)
    class_weights = np.array(class_weights)

    if save_path:
        np.save(save_path, class_weights)

    stats = {
        'cnts': z,
        'freqs': freqs,
        'weights': class_weights
    }
    pprint(stats)

    return class_weights
