# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Custom batch collate functions to pass into pytorch DataLoader
"""
import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern


def collate_any(batch, key=None):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_any([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_any([d[key] for d in batch], key=key) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_any(samples) for samples in zip(*batch)))
    elif key is not None:
        # we are not at top level, so NO TRANSPOSITION!!!
        # unchanged batches for example for keypoints, object_infos
        return batch
    elif isinstance(elem, container_abcs.Sequence):
        assert key is None, "You are transposing something at no-top level, which is WRONG"
        transposed = zip(*batch)
        return [collate_any(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_collate_fn(collate_fn_str):
    if collate_fn_str == "collate_any":
        return collate_any
    return collate_fn_str
