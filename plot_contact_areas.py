import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
from IPython.core.debugger import set_trace
osp = os.path


def plot_contact_areas(instruction):
  bimanual_objects = getattr(utils, '{:s}_bimanual_objects'.format(instruction))
  
  filename = osp.join('data', '{:s}_contact_areas.pkl'.format(instruction))
  with open(filename, 'rb') as f:
    d = pickle.load(f)
  
  fingertips_area = d['contact_areas'][-1]
  
  d = {d['object_names'][i]: d['contact_areas'][i]
      for i in range(len(d['object_names']))}
  data    = {k: v for k,v in d.items() if (k not in bimanual_objects)
      and ('fingertips' not in k)}
  bi_data = {k: v for k,v in d.items() if k in bimanual_objects}
  
  x = np.arange(len(data) + len(bi_data))
  plt.bar(x, list(data.values()) + list(bi_data.values()))
  plt.plot(np.arange(len(data)), fingertips_area*np.ones(len(data)), c='r')
  plt.plot(len(data)+np.arange(len(bi_data)), 2*fingertips_area*np.ones(len(bi_data)), c='r')
  plt.xticks(x, list(data.keys()) + list(bi_data.keys()), rotation=90, fontsize=20)
  plt.ylabel('Contact Area (sq. cm.)', fontsize=20)
  # plt.legend(fontsize=20)
  # plt.title('Contact Area, functional intent = {:s}'.format(instruction))
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  instruction = 'handoff'
  plot_contact_areas(instruction)
