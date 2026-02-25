import h5py
import numpy as np

def set_default_attrs(ds, override=None):
    """
    Set default attributes for an HDF5 dataset.
    
    Args:
        ds: HDF5 dataset to set attributes for.
        override: Dictionary of attributes to override the defaults.
    """
    attrs = {
        'is_list': np.False_,
        'is_ndarray': np.False_,
        'is_str': np.False_,
        'is_tuple': np.False_,
    }
    if override:
        attrs.update(override)
    for k, v in attrs.items():
        ds.attrs[k] = v

def create_true_model(model_path = './Ultrasound-Vp-axial-models/vp_996782.npy', is_data=False, model_data=None, title='BrainTrueModel.h5'):
    """
    Create a true model for the Brain Ultrasound dataset and save it as an HDF5 file.

    Args:
        model_path: Path to the .npy file containing the model data.
        is_data: If True, use the provided model_data instead of loading from model_path.
        model_data: Numpy array containing the model data if is_data is True.
        title: Name of the output HDF5 file.
    """
    # Load the model data
    if is_data:
        model_data = model_data
    else:
        model_data = np.load(model_path)
    shape = model_data.shape  # (320, 256)

    # define space parameters
    extra = np.array([50, 50], dtype='int64')
    absorbing = np.array([40, 40], dtype='int64')
    spacing = np.array([0.0005, 0.0005], dtype='float64')
    extended_shape = np.array([shape[0] + extra[0]*2, shape[1] + extra[1]*2], dtype='int64')

    # inner array
    inner = np.array([
        [b'50', bytes(str(shape[0] + 50), 'utf-8'), b'None'],
        [b'50', bytes(str(shape[1] + 50), 'utf-8'), b'None']
    ])

    # define time parameters
    time_info = {
        'num': 2500,
        'start': 0.0,
        'step': 8e-8,
        'stop': 2500 * 8e-8
    }

    # build the h5 file structure
    with h5py.File(f'{title}', 'w') as f:
        dataset = f.create_dataset('data', data=model_data)
        set_default_attrs(dataset, override={'is_ndarray': np.True_})
        ddtype = f.create_dataset('dtype', data=np.bytes_('float64'))
        set_default_attrs(ddtype, override={'is_str': np.True_})
        dextend = f.create_dataset('extended_shape', data=extended_shape)
        set_default_attrs(dextend, override={'is_tuple': np.True_})
        d_inner = f.create_dataset('inner', data=inner)
        set_default_attrs(d_inner, override={'is_list': np.True_, 'is_str': np.True_})
        dshape = f.create_dataset('shape', data=np.array(shape, dtype='int64'))
        set_default_attrs(dshape, override={'is_tuple': np.True_})
        # group: space/
        space_grp = f.create_group('space')
        dabsorb = space_grp.create_dataset('absorbing', data=absorbing)
        set_default_attrs(dabsorb, override={'is_tuple': np.True_})
        dextra = space_grp.create_dataset('extra', data=extra)
        set_default_attrs(dextra, override={'is_tuple': np.True_})
        dshape = space_grp.create_dataset('shape', data=np.array(shape, dtype='int64'))
        set_default_attrs(dshape, override={'is_tuple': np.True_})
        dspacing = space_grp.create_dataset('spacing', data=spacing)
        set_default_attrs(dspacing, override={'is_tuple': np.True_})

        # group: time/
        time_grp = f.create_group('time')
        dnum = time_grp.create_dataset('num', data=time_info['num'])
        set_default_attrs(dnum)
        dstart = time_grp.create_dataset('start', data=time_info['start'])
        set_default_attrs(dstart)
        dstep = time_grp.create_dataset('step', data=time_info['step'])
        set_default_attrs(dstep)
        dstop = time_grp.create_dataset('stop', data=time_info['stop'])
        set_default_attrs(dstop)

        # time_dependent
        ddependent = f.create_dataset('time_dependent', data=False)
        set_default_attrs(ddependent)