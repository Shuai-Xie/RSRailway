railway_train_stats = {
    1: 193,
    2: 184,
    3: 166,
    4: 109,
    5: 61,
    6: 57,
    7: 38,
    8: 38,
    9: 39,
    10: 32,
    11: 17,
    12: 16,
    13: 16,
    14: 6,
    15: 8,
    16: 8,
    19: 3,
    20: 1,
    21: 3,
    22: 3,
    23: 1,
    24: 1,
    33: 1
}
import matplotlib.pyplot as plt

xs = list(railway_train_stats.keys())
ys = list(railway_train_stats.values())

plt.figure(figsize=(10, 3))
plt.plot(xs, ys)
plt.xlabel('objs')
plt.ylabel('imgs')
plt.title('Railway objs: 1-33, total: {}'.format(sum(ys)))
plt.show()
