import numpy as np
import tqdm

def linear_application(iterator, hdf5_file, function_definitions, append=True):
    """Applicator for function definitions.

    Parameters:
    iterator (list like): An iterator whose elements will be the target
                          of the functions defined in the function
                          definitions.
    hd5f_file (hdf5.File): A hdf5 file handle that will be used to store the
                           resulting analyses
    function_definitions (list): A list containing tuples that define which
                                 functions will be run on the iterable elements
                                 while also providing a way to define the
                                 datatype that will be used in the HDF5 dataset
    """

    length = len(iterator)
    locs = []

    for func, params in function_definitions:
        dataset_name, dataset_dtype, function_kwargs = params
        data = np.zeros((length), dtype=dataset_dtype)

        exists = dataset_name in hdf5_file

        if exists and append:
            print("Appending to dataset: {0}".format(dataset_name))
            locs.append(hdf5_file[dataset_name].attrs["processed"])

            if hdf5_file[dataset_name].shape[0] < length:
                try:
                    print("Attempting to resize")
                    hdf5_file[dataset_name].resize(length, axis=0)
                except Exception as e:
                    print("Couldn't resize dataset")
                    print(e.message)
                    raise e
        elif exists and not append:
            print("Writing over old dataset: {0}".format(dataset_name))
            del hd5f_file[dataset_name]
            locs.append(0)
            hdf5_file.create_dataset(dataset_name, data=data,
                                     chunks=True,
                                     maxshape=((None,) + data.shape[1:]))
            hdf5_file[dataset_name].attrs["processed"] = 0

        else:
            print("Creating new dataset: {0}".format(dataset_name))
            locs.append(0)
            hdf5_file.create_dataset(dataset_name, data=data, chunks=True, maxshape=((None,) + data.shape[1:]))
            hdf5_file[dataset_name].attrs["processed"] = 0

        min_value = min(locs)

    for num, i in tqdm.tqdm(enumerate(iterator[min_value:]), initial=min_value, total=length):
        for func, params in function_definitions:
            number = num + min_value
            dataset_name, dataset_dtype, function_kwargs = params
            dataset = hdf5_file[dataset_name]
            if dataset.attrs["processed"] > (number):
                continue
            try:
                dataset[number] = func(i, **function_kwargs)
            except Exception as e:
                print("\003[31mEncountered error while running: {}\003[0m".format(dataset_name))
                raise e

            dataset.attrs["processed"] = number
