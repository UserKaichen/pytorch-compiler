import csv
import os
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


@dataclass
class Record:
    total_energy: float


def main():
    for filename in os.listdir(r'csv/raw4/output'):
        data = pd.read_csv(f'./csv/raw4/output/{filename}')
        a = ['Chiplet', 'Core', 'Lane', 'Vector_Size', 'AL1', 'real_WL1', 'AL2']
        c = ['note']
        d = ['total_energy']
        e = ['chiplet_communication']
        f = ['dram_communication']

        with open(f'csv/cooked4/{filename}', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['chiplet_communication', 'dram_communication', 'runtime', 'total_energy'])
            writer.writeheader()

            # for design, dt in data.groupby(c):
            #     print(dt)
            #     best_case = dt.sort_values(d)
            #     print(best_case)

            # quit()
            # for design, g1 in data.groupby(a):

            data.sort_values(d).groupby(by=c).first().to_csv(f"./csv/sort/reuslt_{filename}")
            quit()
            # print(data)
            best_design = data.groupby(by=c).sort_values(d).first()
            print(best_design)
            # print(design)

            total_energy = best_design['total_energy']
            chiplet_communication = best_design['chiplet_communication']
            dram_communication = best_design['dram_communication']
            runtime = best_design['runtime']
            # if total_energy == 4544466244.080846:
            #     print('=================\n      pass\n=================')
            # else:
            #     print('=================\n      fail\n=================')
            #     print(total_energy)
            # os._exit(0)
            # total_memory_footprint = best_design['total_memory_footprint'][0].item()

            writer.writerow({'chiplet_communication': chiplet_communication,
                                'dram_communication': dram_communication,
                                'runtime': runtime,
                                'total_energy': total_energy,
                                })


def draw():
    for filename in os.listdir(r'csv/cooked4/'):
        plt.cla()
        plt.close('all')
        data = pd.read_csv(f'./csv/cooked4/{filename}')
        x = data['total_memory_footprint']
        y = data['total_energy']
        plt.scatter(x, y)
        plt.colorbar()
        plt.savefig(f'./csv/cooked4/{filename}.jpg')


if __name__ == '__main__':
    # draw()
    main()
