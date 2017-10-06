import csv
import numpy as np
import matplotlib.pyplot as plt

N_vec = []
t_avg = []
t_std = []
n_avg = []
n_std = []
for N in range(4, 14):
    filename = 'data/nmpc_N' + str(N) + '.csv'
    with open(filename, 'r', newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=' ')
        t = []
        n = []
        for i, r in enumerate(rdr):
            if i % 2 == 0 and i != 0:
                t.append(float(r[8]))
                n.append(float(r[9]))
        N_vec.append(N)
        t_avg.append(np.mean(t))
        t_std.append(np.std(t))
        n_avg.append(np.mean(n))
        n_std.append(np.std(n))

plt.bar(N_vec, t_avg, yerr=t_std)
plt.plot([3, 14], [0.1, 0.1], color='r', linestyle='--')
plt.show()

with open('data/nmpc_timecalc.csv', 'w+', newline='') as csvfile:
    wrt = csv.writer(csvfile, delimiter=' ')
    wrt.writerow(['N', 't_avg', 't_std', 'n_avg', 'n_std'])
    for r in zip(N_vec, t_avg, t_std, n_avg, n_std):
        wrt.writerow(r)
