import numpy as np
import collections
import scipy
import numbers


def __get_slice(volume, slice_index, dimension):

    if dimension == 0:
        out_slice = volume.dataobj[slice_index, :, :]
    elif dimension == 1:
        out_slice = volume.dataobj[:, slice_index, :]
    elif dimension == 2:
        out_slice = volume.dataobj[:, :, slice_index]
    return np.asarray(out_slice)


def __get_slices(volume, start_index, end_index, dimension):

    if dimension == 0:
        out_slices = volume.dataobj[start_index:end_index+1, :, :]
        out_slices = np.transpose(out_slices, axes=[1, 2, 0])
    elif dimension ==1:
        out_slices = volume.dataobj[:, start_index:end_index+1, :]
        out_slices = np.transpose(out_slices, axes = [0,2,1])
    elif dimension ==2:
        out_slices = volume.dataobj[:, :, start_index:end_index+1]

    return np.asarray(out_slices)

def load_slice_filtered(source, label_of_interest=2, label_required=1, slice_type='axial', depth=5, single_label_slice=None, dtype=np.float32):
    if slice_type == 'coronal':
        dimension = 0
    elif slice_type == 'sagittal':
        dimension = 1
    elif slice_type == 'axial':
        dimension = 2
    elif slice_type is None:
        dimension = -1
    else:
        raise TypeError("Unknow slice_type: {}".format(slice_type))

    if depth < 1 or int((depth - 1) / 2) * 2 + 1 != depth:
        raise ValueError("Depth must be a positive integer, Is: {}".format(depth))
    radius = (depth - 1) / 2
    counter = 0
    total = 0
    for inputs in source:
        label_volume = nibable.load(inputs[1])
        min_ = radius
        max_ = label_volume.header.get-data_shape()[dimension] - radius
        indices = np.random.permutation(np.arange(min_, max_))
        indices = [int(i) for i in indices]
        i = 0
        found = False

        if total == 0 or counter / float(total) < min_frequency:
            for i in xrange(min(max_tries, len(indices))):
                label_slice = __get_slice(label_volume,indices[i], dimension)
                if label_required in label_slice and label_of_interest  in label_slice:
                    if depth > 1 and not single_label_slice:
                        label_slice = __get_slices(label_volume, indices[i]-radius, indices[i]+radius, dimension)
                    found = True
                    counter +=1
                    break
        if not found:
            for i in xrange(i, len(indices)):
                label_slice = __get_slice(label_slice, indices[i], dimension)
                if label_required in label_slice:
                    found = True
                    if label_of_interest in label_slice:
                        counter + =1
                    if depth > 1 and not single_label_slice:
                        label_slice = __get_slices(label_volume, indices[i]-radius, indices[i]+radius, dimension)
                        break
        if not found:
            continue

        total += 1
        image_volume = nibabel.load(inputs[0])
        outputs = []
        if depth > 1:
            outputs.append(__get_slices(image_volume, indices[i] - radius, indices[i]+radius, dimension).astype(dtype))
        else:
            outputs.append(__get_slice(image_volume, indices[i],dimension).astype(dtype))
        outputs.append(label_slice.astype(dtype))
        yield outputs

def random_rotation(source, probability=1.0, upper_bound=180):
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("Upper bound must be a number! Received: {}".format(type(upper_bound)))

    if upper_bound < 0:
        raise ValueError("Upper bound must be greater than 0! Received: {}".format(upper_bound))
    elif upper_bound >180:
        upper_bound = 180

    for inputs in source:
        if(np.random.rand() < probability):
            angle = np.random.randint(-upper_bound, upper_bound)
            angle = (360 + angle)%360
            inputs[0] = scipy.ndimage.interpolation.rotate(inputs[0], angle, reshape=False, order=1, cval=np.min(inputs[0]), prefilter=False)
            inputs[1] = scipy.ndimage.interpolation.rotate(inputs[1], angle, reshape=False, order=0, cval=np.min(inputs[1]), prefilter=False)
        yield inputs

def __resize(inputs, desired_size):
    zooms = desired_size / np.array(inputs[0].shape[0:2], dtype=np.float)
    image_zooms = np.append(zooms, np.ones(len(inputs[0].shape)-2, dtype= zooms.dtype))
    inputs[0] = scipy.ndimage.zoom(inputs[0], image_zooms, order=1)  #order = 1  ==>bilinear interpolation

    labels_zooms = np.append(zooms, np.ones(len(inputs[1].shape)-2),dtype = zooms.dtype)
    inputs[1] = scipy.ndimage.zoom(inputs[1], image_zooms, order=0)
    return inputs




def __random_resize(inputs, desired_size, factor, default_pixel, default_label):
    if factor > 1:
        zooms = (desired_size * factor) / np.array(inputs[0].shape[:2], dtype=np.float)
        scaled_size = np.rint(desired_size * factor).astype(np.int)

        x_start = (scaled_size[0] - desired_size[0]) / 2
        y_start = (scaled_size[1] - desired_size[1]) / 2
        x_end = x_start + desired_size[0]
        y_end = y_start + desired_size[1]

        new_size = list(desired_size)
        new_size.extend(inputs[0].shape[2:])
        image_zooms = np.append(zooms, np.ones(len(inputs[0].shape)-2, dtype=zooms.dtype))
        output = np.zeros(new_size, dtype=inputs[0].dtype)
        image = scipy.ndimage.zoom(inputs[0], image_zooms, order=1) #order = 1 =>biliniear interpolation
        output[:, :, ...] = image[x_start,:x_end, y_start: y_end, ...]
        inputs[0] = output

        new_size = list(desired_size)
        new_size.extend(inputs[1].shape[2:])
        label_zooms = np.append(zooms, np.ones(len(inputs[1].shape)-2, dtype=zooms.dtype))

        output = np.zeros(new_size, dtype=zooms.dtype)
        labels = scipy.ndimage.zoom(inputs[1], label_zooms, order = 0)  # order = 0 =>nearest neighbour
        output[:, :, ...] = labels[x_start:x_end, y_start:y_end, ...]
        inputs[1] = output

    elif factor < 1.:
        zooms = (desired_size * factor)/np.array(inputs[0].shape[0:2], dtype=np.float)
        scaled_size = np.rint(desired_size * factor).astype(np.int)

        x_start = (desired_size[0] - scaled_size[0])/2
        y_start = (desired_size[1] - scaled_size[1])/2
        x_end = x_start + desired_size[0]
        y_end = y_start + desired_size[1]

        new_size = list(desired_size)
        new_size.extend(inputs[0].shape[2：])
        image_zooms = np.append(zooms, np.ones(len(inputs[0].shape)-2, dtype=zooms.dtype))

        full_value = default_pixel if default_pixel is not None else np.min(inputs[0])
        output = np.full(new_size, fill_value, dtype=inputs[0].dtype)
        image = scipy.ndimage.zoom(inputs[0], image_zooms, order=1)
        output[x_start:x_end, y_start:y_end, ...] = image[:, :, ...]
        inputs[0] = output

        new_size = list(desired_size)
        new_size.extend(inputs[1].shape[2:])
        labels_zooms = np.append(zooms, np.ones(len(inputs[0].shape)-2, dtype=zooms.dtype))

        fill_value = default_label if default_label is not None else np.min(inputs[1])
        output = np.full(new_size, fill_value, dtype=inputs[1].dtype)
        labels = scipy.ndimage.zoom(inputs[1], label_zooms, order=0)
        output[x_start:x_end, y_start:y_end, ...] = label_zooms[:, :, ...]
        inputs[1] = output

    else:
        inputs = __resize(inputs, desired_size)
    return inputs


def random_resize(source, desired_size, probability = 1.0, lower_bound=0.9, upper_bound=1.1, default_pixel=None, default_label=None):

    if not isinstance(desired_size, collections.Sequence):
        raise TypeError("Desired size must be a sequence or array! Received: {}".format(type(desired_size)))
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))
    if not isinstance(lower_bound,numbers.Number):
        raise TypeError("lower bound must be a number! Received: {}".format(type(lower_bound)))
    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("upper bound must be a number! Received: {}".format(type(upper_bound)))
    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))
    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(default_label))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))
    if upper_bound <= lower_bound:
        raise ValueError("Upper bound must be greater than lower bound! Received: lower:{}, and upper:{}".format(lower_bound,upper_bound))
    if lower_bound <= 0.0:
    raise ValueError("Lower bound must be greater than 0.0! Received: {}".format(lower_bound))

    desired_size = np.asarray(desired_size, dtype=np.int)
    for inputs in source:
        if(np.random.rand()<probability):
            factor = np.random.rand()*(upper_bound - lower_bound)+lower_bound
            inputs = __random_resize(inputs, desired_size, factor, default_pixel, default_label)
        else:
            inputs = __resize(inputs, inputs)

        yield inputs

def random_translation(source, probability=0.5, border_usage=0.5, default_border=0.25, label_of_interest=1, default_pixel=-100, default_label=0):

    if not isinstance(porbability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(border_usage, numbers.Number):
        raise TypeError("Border usage must be a number! Receive: {}".format(type(border_usage)))

    if not isinstance(default_border, numbers.Number):
        raise TypeError("Default border must be a number! Received: {}".format(type(default_border)))

    if label_of_interest is not None and not isinstance(label_of_interest, numbers.Number):
        raise TypeError("Label of interest must be a number! Receive: {}".format(type(label_of_interest)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not NOne and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    if border_usage > 1.0 or border_usage < 0.0:
        raise ValueError("Border usage must be between 0.0 and 1.0! Received: {}".format(border_usage))

    if default_border > 1.0 or default_border < 0.0:
        raise ValueError("Default border must be between 0.0 and 1.0! Received: {}".format(default_border))

    def non(s):
        return s if s < 0 else None

    def mom(s):
        return max(0, s)

    for inputs in source:
        if np.random.rand() < probability:
            img = inputs[0]
            label = inputs[1]

            if label_of_interest is None or label_of_interest not in label:
                xdist = default_border * label.shape[0]
                ydist = default_border * label.shape[1]

            else:
                itemindex = np.where(label==1)
                xdist = min(np.min(itemindex[0]), label.shape[0] - np.max(itemindex[0])) * border_usage
                ydist = min(np.min(itemindex[0]), label.shape[1] - np.max(itemindex[1])) * border_usage

            ox = np.random.randint(-xdist, xdist) if xdist > 1 else 0
            oy = np.random.randint(-ydist, ydist) if ydist > 1 else 0

            fill_value = default_pixel if default_pixel is not None else np.min(img)
            shift_img = np.full_like(img, fill_value)
            shift_img[mom(ox):non(ox), mom(oy):non(oy), ...] = img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]
            inputs[0] = shift_img

            fill_value = default_label if default_label is not None else np.min(label)
            shift_lable = np.full_like(label,fill_value)
            shift_lable[mom(ox):non(ox), mom(oy):non(oy), ...] = label[mom(-ox):non(-ox), mom(oy):non(-oy), ... ]
            inputs[1] = shift_lable

            yield inputs



def reduce_to_single_label_slice(source):
    for inputs in source:
        labels = inputs[1]
        while len(labels.shape) > 2:
            index = (labels.shape[-1] -1) /2
            labels = labels[..., index]
        inputs[1] = labels
        yield inputs

def clip_img_values(node, minimum, maximum):

    if not isinstance(minimum, numbers.Number):
        raise TypeError("Minimum must be a number! Received: {}".format(type(minimum)))

    if not isinstance(maximum, numbers.Number):
        raise TypeError("Maximum must be a number! Received: {}".format(type(maximum)))

    for inputs in source:
        np.clip(inputs[0], minimum, maximum, out=inputs[0])
        yield inputs


def fuse_labels_greater_than(source, threshold):
    if not isinstance(threshold, numbers.Number):
        for inputs in source:
            inputs[1] = (inputs[1]>threshold).astype(inputs[1].dtype)
            yield inputs


def transpose(source):
    for inputs in source:
        if len(inputs[0].shape) > 2:
            axes = np.arange(len(inputs[0].shape))
            axes[0] = 1
            axes[1] = 0
            inputs[0] = np.transpose(inputs[0], axes)
        else:
            inputs[0] = np.transpose(inputs[0])


        if len(inputs[1].shape) > 2:
            axes = np.arange(len(inputs[0].shape))
            axes[0] = 1
            axes[1] = 0
            inputs[1] = np.transpose(inputs[1], axes)
        else:
            inputs[1] = np.transpose(inputs[1])
        yiled inputs


def robust_img_scaling(source, ignore_values=[], initialization_generator=None, initialize=True):

    import sklearn.preprocessing as skpre

    scaler = skpre.RobustScaler(copy=False)

    if initialize:
        dictionary = {}

    for inputs in source:
        image = inputs[0]

        old_shape = image.shape
        image = image.reshape(-1,1)

        if len(ignore_values) > 0:
            img_fit = image
            for value in ignore_values:
                img_fit = np.ma.masked_values(img_fit, value, copy=False)
            img_fit = np.ma.compressed(img_fit)
            img_fit = img_fit.reshape(-1,1)
            scaler.fit(img_fit)

        else:
            scaler.fit(image)

        image = scaler.transform(image)
        inputs[0] = image.reshape(old_shape)
        yield inputs










