import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.core import MultiContainerInterface, DynamicTable

from ndbox.utils import get_root_logger, dict2str, split_string
from ndbox.utils.registry import DATA_REGISTRY


@DATA_REGISTRY.register()
class NWBDataset:
    """
    A class for loading data from NWB files.
    """

    def __init__(self, opt):
        """
        Initializes an NWBDataset, loading data from
        the indicated file(s).

        :param opt: dict. Config for datasets. It contains the following keys:
            path - str. The path to an NWB file.
            skip_fields - str, optional. List of field names to skip during loading,
                which may be useful if memory is an issue. Field names must match the
                names automatically assigned in the loading process. Spiking data
                can not be skipped. Field names in the list that are not found in the
                dataset are ignored. The field names are separated by symbols ',' or ';',
                such as "units, images, events".
        """

        self.opt = opt
        self.units_opt = opt.get('units', {})
        self.trials_opt = opt.get('trials', {})
        self.behavior_list = opt.get('behavior', [])
        name = opt.get('name', 'NWBDataset')
        path = opt['path']
        self.logger = get_root_logger()
        self.logger.info(f"Loading dataset '{name}' from file '{path}'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' File not found!")
        if os.path.isdir(path):
            raise FileNotFoundError(f"The specified path '{path}' is a "
                                    f"directory， requires a file")
        self.filename = path
        skip_str = opt.get('skip_fields')
        self.skip_fields = (split_string(skip_str, ',|;') if skip_str is not None else [])
        self.data_dict, self.content_dict = self._build_data()
        self.data = {}
        # self._load_data()

    def _build_data(self):
        io = NWBHDF5IO(self.filename, 'r')
        nwb_file = io.read()
        data_dict, content_dict = self._find_items(nwb_file, '')
        return data_dict, content_dict

    def _find_items(self, obj, prefix):
        data_dict = {}
        content_dict = {}
        for child in obj.children:
            name = prefix + child.name
            if name in self.skip_fields:
                continue
            content_dict[child.name] = {}
            if isinstance(child, MultiContainerInterface):
                d1, d2 = self._find_items(child, name + '/')
                data_dict.update(d1)
                content_dict[child.name] = d2
            else:
                data_dict[name] = child
        return data_dict, content_dict

    def content_repr(self):
        return dict2str(self.content_dict)

    def data_info(self):
        msg = ''
        for val in self.data_dict.values():
            msg += f'{val}\n'
        return msg

    def make_data(self, field):
        obj = self.data_dict.get(field)
        if obj is None:
            raise KeyError(f"No field '{field}' in data dict!")
        elif isinstance(obj, TimeSeries):
            if obj.timestamps is not None:
                time_index = obj.timestamps[()]
            else:
                time_index = np.arange(obj.data.shape[0]) / obj.rate + obj.starting_time
            index = pd.to_timedelta(time_index.round(6), unit='s')
            columns = []
            if len(obj.data.shape) == 2:
                for i in range(obj.data.shape[1]):
                    columns.append(obj.name + '_' + str(i))
            elif len(obj.data.shape) == 1:
                columns.append(obj.name)
            else:
                self.logger.warning(f"Not support data dims larger than 2, "
                                    f"'{obj.name}' shape is '{obj.data.shape}'")
                return obj.data[()]
            df = pd.DataFrame(obj.data[()], index=index, columns=columns)
            return df
        elif isinstance(obj, DynamicTable):
            df = obj.to_dataframe()
            return df
        self.logger.warning(f"The '{obj.name}' class '{obj.__class__.__name__}' not support!")
        return None

    def _load_data(self):
        start_time = 0
        stop_time = 0
        bin_width = self.units_opt.get('bin_width', 1)

        behavior_data_dict = {}
        for behavior_field in self.behavior_list:
            behavior_data = self.make_data(behavior_field)
            if len(behavior_data.shape) > 2:
                continue
            end_time = round(float(behavior_data.index[-1] / np.timedelta64(1, 's')), 6)
            stop_time = max(stop_time, end_time)
            behavior_data_dict[behavior_field] = behavior_data

        trials_field = self.trials_opt.get('field', 'trials')
        trials_obj = self.data_dict.get(trials_field)
        if trials_obj is None:
            self.logger.warning("Trials data not found in dataset! Default "
                                "define trials to be 600 ms segment.")
        else:
            trials = trials_obj.to_dataframe()
            for name in trials.colnames:
                if not hasattr(trials, name):
                    self.logger.warning(f"Field '{name}' not found in NWB file trials table!")

        units_field = self.units_opt.get('field')
        if units_field is None:
            if 'Units' in self.content_dict:
                units_field = 'Units'
            elif 'units' in self.content_dict:
                units_field = 'units'
            else:
                self.logger.warning("Units data not found in dataset!")
                return
        units_obj = self.data_dict.get(units_field)
        if isinstance(units_obj, DynamicTable):
            units = units_obj.to_dataframe()
            if 'spike_times' not in units.columns:
                self.logger.warning("Spike times data not found in units!")
                return
            if 'obs_intervals' in units.columns:
                end_time = max(units.obs_intervals.apply(lambda x: x[-1][-1]))
                stop_time = max(stop_time, end_time)
        elif isinstance(units_obj, TimeSeries):
            if units_obj.rate is None:
                self.logger.warning(f"The units is a Timeseries class, "
                                    f"the rate attribute must be included.")
                return
            timestamps = np.arange(units_obj.data.shape)
        else:
            self.logger.warning(f"Units data class not support!")
            return

