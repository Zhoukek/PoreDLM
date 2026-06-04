"""
Bonito Fast5 Utils
"""

import sys
from glob import glob
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing import Pool
from datetime import timedelta, timezone

import numpy as np
import bonito.reader
from tqdm import tqdm
from dateutil import parser
from scipy.ndimage import median_filter
from ont_fast5_api.fast5_interface import get_fast5_file


def get_attr(attrs, key, default='unset', encoding='ascii'):
    value = attrs.get(key, default)
    if type(value) in (bytes, np.bytes_):
        value = value.decode(encoding)
    return value


def nanopore_normalize_mongo(signal):
    signal_clear = nanopore_repair_errors(signal, 1, 220)
    signal_elite = nanopore_remove_spikes(
        signal_clear, window_size=6000, spike_threshold=5.0
    )
    signal_normal, scale = nanopore_normalize_novel(signal_elite)
    return signal_normal, scale


def nanopore_normalize_novel(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal, 1.0

    center = np.median(signal)
    residual = signal - center
    q01, q99 = np.quantile(residual, [0.01, 0.99])
    masked_residual = residual[(residual >= q01) & (residual <= q99)]

    scale = 1.4826 * np.median(np.abs(masked_residual))
    scale = max(scale, 1.0)
    return (residual / scale).astype(np.float32), scale


def nanopore_repair_errors(signal, min_value, max_value):
    signal = np.asarray(signal, dtype=np.float32)
    valid_mask = (signal >= min_value) & (signal <= max_value)
    if signal.size == 0 or np.all(valid_mask):
        return signal

    cleaned = signal.copy()
    for i in np.where(~valid_mask)[0]:
        if i == 0:
            cleaned[i] = max_value if cleaned[i] > max_value else min_value
        else:
            cleaned[i] = cleaned[i - 1]
    return cleaned


def nanopore_remove_spikes(signal, window_size=5001, spike_threshold=5.0):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal

    local_med = median_filter(signal, size=window_size, mode='reflect')
    residual = signal - local_med
    global_mad = 1.4826 * np.median(np.abs(residual))
    global_mad = max(global_mad, 1.0)
    is_spike = np.abs(residual) > (spike_threshold * global_mad)
    if not np.any(is_spike):
        return signal.copy()

    cleaned = signal.copy()
    for i in np.where(is_spike)[0]:
        cleaned[i] = local_med[i] if i == 0 else cleaned[i - 1]
    return cleaned


class Read(bonito.reader.Read):

    def __init__(self, read, filename, meta=False, do_trim=True, scaling_strategy=None, norm_params=None):

        self.meta = meta
        self.read_id = read.read_id
        self.filename = filename.name
        self.run_id = read.get_run_id()
        if type(self.run_id) in (bytes, np.bytes_):
            self.run_id = self.run_id.decode('ascii')

        tracking_id = read.handle[read.global_key + 'tracking_id'].attrs

        self.sample_id = get_attr(tracking_id, 'sample_id')

        self.exp_start_time = get_attr(
            tracking_id, 'exp_start_time', default='1970-01-01T00:00:00'
        )
        self.exp_start_time = self.exp_start_time.replace('Z', '')

        self.flow_cell_id = get_attr(tracking_id, 'flow_cell_id')

        self.device_id = get_attr(tracking_id, 'device_id')

        if self.meta:
            return
        
        # 原始版本
        # read_attrs = read.handle[read.raw_dataset_group_name].attrs
        # channel_info = read.handle[read.global_key + 'channel_id'].attrs

        # self.offset = int(channel_info['offset'])
        # self.sample_rate = channel_info['sampling_rate']
        # self.scaling = channel_info['range'] / channel_info['digitisation']

        # self.mux = read_attrs['start_mux']
        # self.read_number = read_attrs['read_number']
        # self.channel = channel_info['channel_number']
        # if type(self.channel) in (bytes, np.bytes_):
        #     self.channel = self.channel.decode()

        # self.start = read_attrs['start_time'] / self.sample_rate
        # self.duration = read_attrs['duration'] / self.sample_rate

        # exp_start_dt = parser.parse(self.exp_start_time)
        # start_time = exp_start_dt + timedelta(seconds=self.start)
        # self.start_time = start_time.astimezone(timezone.utc).isoformat(timespec="milliseconds")

        # raw = read.handle[read.raw_dataset_name][:]
        # self.scaled = np.array(self.scaling * (raw + self.offset), dtype=np.float32)
        # self.num_samples = len(self.scaled)

        # self.scaling_strategy = ("quantile" if scaling_strategy is None else
        #                          scaling_strategy.get("strategy","quantile")) 
        # self.shift, self.scale = bonito.reader.normalisation(self.scaled, scaling_strategy, norm_params)
        # self.trimmed_samples = bonito.reader.trim(self.scaled, threshold=self.scale * 2.4 + self.shift) if do_trim else 0
        # self.template_start = self.start + (self.trimmed_samples / self.sample_rate)
        # self.template_duration = self.duration - (self.trimmed_samples / self.sample_rate)

        # self.signal = (self.scaled[self.trimmed_samples:] - self.shift) / self.scale

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        self.offset = int(channel_info['offset'])
        self.sample_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']

        self.mux = read_attrs['start_mux']
        self.read_number = read_attrs['read_number']
        self.channel = channel_info['channel_number']
        if type(self.channel) in (bytes, np.bytes_):
            self.channel = self.channel.decode()

        self.start = read_attrs['start_time'] / self.sample_rate
        self.duration = read_attrs['duration'] / self.sample_rate

        exp_start_dt = parser.parse(self.exp_start_time)
        start_time = exp_start_dt + timedelta(seconds=self.start)
        self.start_time = start_time.astimezone(timezone.utc).isoformat(timespec="milliseconds")

        raw = read.handle[read.raw_dataset_name][:]
        self.scaled = np.array(self.scaling * (raw + self.offset), dtype=np.float32)
        self.num_samples = len(self.scaled)

        self.scaling_strategy = "mongo"
        self.trimmed_samples = 0
        self.shift = 0.0
        self.signal, self.scale = nanopore_normalize_mongo(self.scaled)
        self.template_start = self.start + (self.trimmed_samples / self.sample_rate)
        self.template_duration = self.duration - (self.trimmed_samples / self.sample_rate)


def get_meta_data(filename, read_ids=None, skip=False):
    """
    Get the meta data from the fast5 file for a given `filename`.
    """
    meta_reads = []
    with get_fast5_file(filename, 'r') as f5_fh:
        try:
            all_read_ids = f5_fh.get_read_ids()
        except RuntimeError as e:
            sys.stderr.write(f"> warning: f{filename} - {e}\n")
            return meta_reads
        for read_id in all_read_ids:
            if read_ids is None or (read_id in read_ids) ^ skip:
                meta_reads.append(
                    Read(f5_fh.get_read(read_id), filename, meta=True)
                )
        return meta_reads


def get_read_groups(directory, model, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all the read meta data for a given `directory`.
    """
    groups = set()
    num_reads = 0
    pattern = "**/*.fast5" if recursive else "*.fast5"
    fast5s = [Path(x) for x in glob(directory + "/" + pattern, recursive=True)]
    get_filtered_meta_data = partial(get_meta_data, read_ids=read_ids, skip=skip)

    with Pool(n_proc) as pool:
        for reads in tqdm(
                pool.imap(get_filtered_meta_data, fast5s), total=len(fast5s), leave=False,
                desc="> preprocessing reads", unit=" fast5s", ascii=True, ncols=100
        ):
            groups.update({read.readgroup(model) for read in reads})
            num_reads += len(reads)
        return groups, num_reads


def get_read_ids(filename, read_ids=None, skip=False):
    """
    Get all the read_ids from the file `filename`.
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        try:
            ids = [(filename, rid) for rid in f5_fh.get_read_ids()]
        except RuntimeError as e:
            sys.stderr.write(f"> warning: f{filename} - {e}\n")
            return []
        if read_ids is None:
            return ids
        return [rid for rid in ids if (rid[1] in read_ids) ^ skip]


def get_raw_data_for_read(info, do_trim=True, scaling_strategy=None, norm_params=None):
    """
    Get the raw signal from the fast5 file for a given filename, read_id pair
    """
    filename, read_id = info
    with get_fast5_file(filename, 'r') as f5_fh:
        return Read(f5_fh.get_read(read_id), filename, do_trim=do_trim, scaling_strategy=scaling_strategy, norm_params=norm_params)


def get_raw_data(filename, read_ids=None, skip=False):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        for read_id in f5_fh.get_read_ids():
            if read_ids is None or (read_id in read_ids) ^ skip:
                yield Read(f5_fh.get_read(read_id), filename)


def get_reads(directory, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None, do_trim=True, scaling_strategy=None, norm_params=None):
    """
    Get all reads in a given `directory`.
    """
    pattern = "**/*.fast5" if recursive else "*.fast5"
    get_filtered_reads = partial(get_read_ids, read_ids=read_ids, skip=skip)
    get_raw_data = partial(get_raw_data_for_read, do_trim=do_trim, scaling_strategy=scaling_strategy, norm_params=norm_params)
    reads = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))
    with Pool(n_proc) as pool:
        for job in chain(pool.imap(get_filtered_reads, reads)):
            for read in pool.imap(get_raw_data, job):
                yield read
                if cancel is not None and cancel.is_set():
                    return
