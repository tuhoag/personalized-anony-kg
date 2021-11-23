import itertools
import os
import glob

import anonygraph.utils.path as putils

data_name = "freebase"
sample = -1

anony_clusters_dir_path = putils.get_clusters_dir_path(data_name, sample, "anony")
raw_clusters_dir_path = putils.get_clusters_dir_path(data_name, sample, "raw")

pattern = "rtd"
new_pattern = "te"

anony_clusters_paths = glob.glob(os.path.join(anony_clusters_dir_path, "*{}*.txt".format(pattern)))
raw_clusters_paths = glob.glob(os.path.join(raw_clusters_dir_path, "*{}*.txt".format(pattern)))
print("found {} anony and {} raw clusters paths".format(len(anony_clusters_paths), len(raw_clusters_paths)))

count = 0
for path in itertools.chain(anony_clusters_paths, raw_clusters_paths):
    count += 1

    dir_name = os.path.dirname(path)
    old_base_name = os.path.basename(path)
    new_base_name = old_base_name.replace(pattern, new_pattern)

    new_path = os.path.join(dir_name, new_base_name)

    print(old_base_name, new_base_name)

    os.rename(path, new_path)

print("replaced {} files".format(count))