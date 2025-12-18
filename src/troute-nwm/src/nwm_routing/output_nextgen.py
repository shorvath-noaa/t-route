import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any
import logging

LOG = logging.getLogger(__name__)

class AbstractOutputWriter(ABC):
    @abstractmethod
    def initialize(self, config: Any, all_network_ids: np.ndarray, 
                   total_sim_seconds: float, start_time: datetime, dt: float):
        pass

    @abstractmethod
    def write_step(self, run_results: List[Tuple], 
                   current_chunk_start_time: datetime):
        pass

    @abstractmethod
    def close(self):
        pass

class NetCDFStreamWriter(AbstractOutputWriter):
    def __init__(self):
        self.nc = None
        self.cfg = None
        self.sorter = None
        self.output_ids = None
        self.total_steps = 0
        self.dt = 300.0  
        self.global_start_time = None 
        
        # Offsets for standard variables in the r[1] array
        self.var_offsets = {
            'streamflow': 0, 'velocity': 1, 'depth': 2
        }

    def initialize(self, config: Any, all_network_ids: np.ndarray, 
                   total_sim_seconds: float, start_time: datetime, dt: float):
        
        self.cfg = config
        self.dt = float(dt)
        self.global_start_time = start_time
        
        output_path = self.cfg['output_path']
        output_interval = self.cfg['output_interval']
        variables = self.cfg['variables']

        LOG.info(f"Initializing NetCDF Stream Output: {output_path}")

        # 1. Calculate Total Output Steps
        self.total_steps = int(total_sim_seconds // output_interval)
        
        # 2. Setup ID Indexing
        self._setup_indexing(all_network_ids)

        # 3. Create NetCDF
        self.nc = nc.Dataset(output_path, "w", format="NETCDF4")
        self.nc.createDimension("time", self.total_steps)
        self.nc.createDimension("feature_id", len(self.output_ids))
        self.nc.createDimension("reference_time", 1)

        # 4. Coordinates
        v_id = self.nc.createVariable("feature_id", "i4", ("feature_id",))
        v_id[:] = self.output_ids

        v_time = self.nc.createVariable("time", "i4", ("time",), fill_value=-9999)
        v_time.units = f"minutes since {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        v_time.calendar = "standard"
        v_time.long_name = "valid output time"
        
        # 5. Data Variables
        # Chunking Strategy: Optimize for writing one full timestep at a time (1 x N)
        chunk_dims = (1, len(self.output_ids))
        comp_args = {'zlib': True, 'complevel': 2, 'fill_value': np.nan, 'chunksizes': chunk_dims}
        
        for var_name in variables:
            self.nc.createVariable(var_name, "f4", ("time", "feature_id"), **comp_args)

    def _setup_indexing(self, all_network_ids: np.ndarray):
        subset_ids = None
        subset_file = self.cfg.get('subset_file')

        if subset_file:
            try:
                with open(subset_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        # Try common keys
                        subset_ids = data.get('flowpath_ids') or data.get('link_ids') or list(data.values())[0]
                    elif isinstance(data, list):
                        subset_ids = data
                    
                    if subset_ids: 
                        subset_ids = np.array(subset_ids)
            except Exception as e:
                LOG.error(f"Failed to read subset file: {e}")
                raise

        if subset_ids is not None:
            valid_subset = np.intersect1d(all_network_ids, subset_ids)
            self.output_ids = np.sort(valid_subset)
        else:
            self.output_ids = np.sort(all_network_ids)
        
        self.sorter = np.argsort(self.output_ids) 

    def write_step(self, run_results: List[Tuple], current_chunk_start_time: datetime):
        if self.nc is None or not run_results: return
        
        # 1. Determine dimensions
        sample_r = run_results[0]
        if sample_r[1] is None: return 
        ncols = sample_r[1].shape[1]
        nts_chunk = ncols // 3
        
        output_interval = self.cfg['output_interval']
        
        # 2. Iterate through internal timesteps
        for i in range(1, nts_chunk + 1):
            step_time = current_chunk_start_time + timedelta(seconds=i * self.dt)
            time_since_start = (step_time - self.global_start_time).total_seconds()
            
            # Check alignment
            if abs(time_since_start % output_interval) < 0.1:
                nc_time_idx = int(round((time_since_start - output_interval) / output_interval))
                
                if 0 <= nc_time_idx < self.total_steps:
                    self._write_single_frame(run_results, step_time, i-1, nc_time_idx)

    def _write_single_frame(self, run_results: List[Tuple], absolute_time: datetime, 
                            step_index: int, nc_time_idx: int):
        
        # Write Time Value
        elapsed_minutes = int((absolute_time - self.global_start_time).total_seconds() / 60)
        self.nc.variables['time'][nc_time_idx] = elapsed_minutes
        
        base_col = step_index * 3
        
        variables = self.cfg['variables']

        # Allocate full domain arrays filled with NaN
        data_buffers = {}
        for var in variables:
            data_buffers[var] = np.full(len(self.output_ids), np.nan, dtype=np.float32)
        
        # Fill buffers
        for r in run_results:
            chunk_ids = r[0]
            if chunk_ids is None or len(chunk_ids) == 0: continue

            # Index Lookup
            idx_in_nc = np.searchsorted(self.output_ids, chunk_ids, sorter=self.sorter)
            
            # Mask Validation
            in_bounds = idx_in_nc < len(self.output_ids)
            valid_mask = np.zeros_like(in_bounds, dtype=bool)
            valid_mask[in_bounds] = (self.output_ids[idx_in_nc[in_bounds]] == chunk_ids[in_bounds])

            if not np.any(valid_mask): continue
            
            target_indices = idx_in_nc[valid_mask]

            # Extract Data
            for var_name in variables:
                if var_name == 'nudge':
                    #TODO: Update to include nudge values. Skipping for now.
                    # Nudge Logic: Usually r[8]. Needs Mapping if Nudge IDs != Seg IDs
                    pass
                elif var_name in self.var_offsets:
                    offset = self.var_offsets[var_name]
                    target_col = base_col + offset
                    
                    # Fill Buffer
                    data_buffers[var_name][target_indices] = r[1][:, target_col][valid_mask]

        # Write to disk
        for var_name, data_array in data_buffers.items():
            self.nc.variables[var_name][nc_time_idx, :] = data_array

    def close(self):
        if self.nc:
            self.nc.sync()
            self.nc.close()
            self.nc = None