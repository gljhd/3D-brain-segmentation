import nibable
import reader
import segmentation as seg
import numpy as np


def Pipeline(dir):
    node = reader.file_path(dir)

    node = seg.load_slice_filtered(node, label_of_interest=1, depth=5)

    node = seg.random_rotation(node, probability=1.0, upper_bound=180)

    node = seg.random_resize(node, [256, 256], probability=1.0, )
    node = seg.random_translation(node, probability=0.5, border_usage=0.5, default_border=0.25, label_of_interest=1, default_pixel=-100, default_label=0)
    node = seg.reduce_to_single_label_slice(node)
    node = seg.clip_img_values(node, -100., 400.)
    node = seg.fuse_labels_greater_than(node, 0.5)

    node = seg.transpose(node)

    node = seg.robust_img_scaling(node, ignore_values=[-100,400], initialize = False)

    return node

def data_generator(batch_size, file_path):
    lines = Pipeline(file_path)
    img = []
    label = []
    t = 0
    for x,y in lines:
        if t == batch_size:
            img = np.array(img).astype(np.float32)
            label = np.array(label).astype(np.int32)
            yield (img, label)
            img = []
            label = []
            t = 0
        else:
            img.append(x)
            label.append(y)
            t = t + 1