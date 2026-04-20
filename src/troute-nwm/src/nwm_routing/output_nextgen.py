import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Optional, Any
import logging

LOG = logging.getLogger("")

_FVD_POSITIONS = {'streamflow': 0, 'velocity': 1, 'depth': 2}
_VAR_ATTRS = {
    'streamflow': ('m3 s-1', 'Streamflow'),
    'velocity':   ('m s-1',  'Velocity'),
    'depth':      ('m',      'Depth'),
    'nudge':      ('m3 s-1', 'Data assimilation nudge applied to streamflow'),
}


# Abstract Class for Output Writing Methods (Stream output, Reservoir output, etc.)
class AbstractOutputWriter(ABC):
    @abstractmethod
    def initialize(self, config: Any, all_network_ids: np.ndarray, 
                   total_sim_seconds: float, start_time: datetime, dt: float,
                   rconn: Dict[int, List[int]]):
        pass

    @abstractmethod
    def write_step(self, run_results: List[Tuple], 
                   current_chunk_start_time: datetime):
        pass

    @abstractmethod
    def close(self):
        pass


# Stream output class
class NetCDFStreamWriter(AbstractOutputWriter):
    def __init__(self):
        self._nc = None
        self._cfg = None
        self._total_steps = 0
        self._dt = 300.0
        self._global_start_time = None
 
        # Ordered arrays of integer IDs and type labels ('wb' or 'nex'),
        # one entry per row in the NetCDF feature_id dimension.
        self._output_ids = None 
        self._output_types = None 

        # Crosswalks
        self._wb_id_to_pos:  Dict[int, int] = {}
        self._nex_id_to_pos: Dict[int, int] = {}
        self._nex_upstream:  Dict[int, List[int]] = {}

    def initialize(self, config: Any, all_network_ids: np.ndarray, 
                   total_sim_seconds: float, start_time: datetime, dt: float,
                   rconn: Dict[int, List[int]]):
        
        self._cfg = config
        self._dt = float(dt)
        self._global_start_time = start_time
        self._total_steps = int(total_sim_seconds // config['output_interval'])
        
        self.build_id_index(all_network_ids, rconn)
        self.create_netcdf(config['output_path'], config['variables'], start_time)

    def write_step(self, run_results: List[Tuple],
                   current_chunk_start_time: datetime):
        if self._nc is None or not run_results:
            return
        
        fvd_sample = run_results[0][1]
        nts_chunk = fvd_sample.shape[1] // 3
        output_interval = self._cfg['output_interval']
        
        for i in range(1, nts_chunk + 1):
            step_time = current_chunk_start_time + timedelta(seconds=i * self._dt)
            elapsed   = (step_time - self._global_start_time).total_seconds()
 
            if abs(elapsed % output_interval) < 0.1:
                nc_idx = int(round(elapsed / output_interval)) - 1
                if 0 <= nc_idx < self._total_steps:
                    self.write_frame(run_results, step_time, i - 1, nc_idx)
 
    def close(self):
        if self._nc:
            self._nc.sync()
            self._nc.close()
            self._nc = None


    # Helper Functions
    def build_id_index(self, all_network_ids: np.ndarray,
                       rconn: Dict[int, List[int]]):
        """
        Parse the subset file (if any) and build all internal lookup
        structures.  Rows are ordered: sorted wb reaches first, then
        sorted nexus rows.
        """
        wb_ids, nex_ids = self.load_subset(all_network_ids)
 
        # Find the upstream connections for each requested nexus.
        all_set = set(all_network_ids.tolist())
        nex_upstream = {}
        
        for nid in sorted(nex_ids):
            upstream = [u for u in rconn.get(nid, []) if u in all_set]
            nex_upstream[nid] = upstream
 
        wb_sorted = np.sort(np.array(list(wb_ids),  dtype=np.int64))
        nex_sorted = np.sort(np.array(list(nex_ids), dtype=np.int64))
 
        self._output_ids = np.concatenate([wb_sorted, nex_sorted])
        self._output_types = np.array(
            ['wb'] * len(wb_sorted) + ['nex'] * len(nex_sorted),
            dtype='U3'
        )
 
        offset = len(wb_sorted)
        self._wb_id_to_pos = {int(fid): i for i, fid in enumerate(wb_sorted)}
        self._nex_id_to_pos = {int(nid): offset + i
                               for i, nid in enumerate(nex_sorted)}
        self._nex_upstream = nex_upstream
        
        self._nex_upstream_ids: frozenset = frozenset(
            uid for upstream in nex_upstream.values() for uid in upstream
        )
 
    def load_subset(self, all_network_ids: np.ndarray
                     ) -> Tuple[Set[int], Set[int]]:
        """
        Return (wb_ids, nex_ids) as sets of integers.
 
        If no subset file is configured, wb_ids contains all network IDs
        and nex_ids is empty.
        """
        subset_file = self._cfg.get('subset_file')
        if not subset_file:
            return set(all_network_ids.tolist()), set()
 
        with open(subset_file) as fh:
            data = yaml.safe_load(fh)

        raw_wb, raw_nex = [], []
        raw_wb  = data.get('feature_ids', [])
        raw_nex = data.get('nexuses', [])

        wb_ids  = self.remove_prefix(raw_wb)
        nex_ids = self.remove_prefix(raw_nex)
 
        all_set = set(all_network_ids.tolist())
        invalid_wb = wb_ids - all_set
        if invalid_wb:
            LOG.warning("%d requested feature_ids not found in network and "
                        "will be skipped", len(invalid_wb))
        wb_ids &= all_set
 
        if not wb_ids and not nex_ids:
            LOG.warning("subset_file produced no usable IDs; "
                        "writing all network IDs as wb reaches")
            return all_set, set()
        
        return wb_ids, nex_ids
 
    @staticmethod
    def remove_prefix(raw: list) -> Set[int]:
        out = set([int(float(s.split('-', 1)[-1])) for s in raw])
        return out
 
    def create_netcdf(self, output_path: str, variables: List[str],
                       start_time: datetime):
        nfeatures = len(self._output_ids)
 
        self._nc = nc.Dataset(output_path, 'w', format='NETCDF4')
        self._nc.createDimension('time', self._total_steps)
        self._nc.createDimension('feature_id', nfeatures)
 
        v_id = self._nc.createVariable('feature_id', 'i4', ('feature_id',))
        v_id.long_name = 'Reach or nexus integer ID'
        # v_id.cf_role = 'timeseries_id' TODO: DELETE
        v_id[:] = self._output_ids.astype(np.int32)
 
        # 'wb' for flowpath reaches, 'nex' for nexus aggregation rows
        v_type = self._nc.createVariable('feature_type', str, ('feature_id',))
        v_type.long_name = ('Feature type: "wb" = flowpath reach, "nex" = nexus point')
        v_type[:] = self._output_types
        
        v_time = self._nc.createVariable('time', 'i4', ('time',), fill_value=-9999)
        v_time.units = 'minutes since {}'.format(start_time.strftime('%Y-%m-%d %H:%M:%S'))
        v_time.calendar = 'standard'
        v_time.long_name = 'valid output time'
 
        chunk_dims = (1, nfeatures)
        comp_args = dict(zlib=True, complevel=2, fill_value=np.nan,
                         chunksizes=chunk_dims)
 
        for var_name in variables:
            units, long_name = _VAR_ATTRS.get(var_name, ('', var_name))
            v = self._nc.createVariable(var_name, 'f4', ('time', 'feature_id'), **comp_args)
            v.units = units
            v.long_name = long_name
 
        self._nc.featureType = 'timeSeries'
        self._nc.Conventions = 'CF-1.8'
 
        LOG.info("NetCDF stream output initialised: %s  "
                 "(%d wb reaches, %d nexus points, %d timesteps)",
                 output_path, len(self._wb_id_to_pos),
                 len(self._nex_id_to_pos), self._total_steps)

    def write_frame(self, run_results: List[Tuple], abs_time: datetime,
                     step_index: int, nc_time_idx: int):
        elapsed_min = int((abs_time - self._global_start_time).total_seconds() / 60)
        self._nc.variables['time'][nc_time_idx] = elapsed_min
 
        base_col = step_index * 3
        variables = self._cfg['variables']
        nout = len(self._output_ids)
 
        output_dict = {v: np.full(nout, np.nan, dtype=np.float32) for v in variables}
        
        # Dictionaries for nexus aggregation: seg_id -> value this timestep.
        seg_flow = {}  # populated when streamflow or nexus rows are needed
        seg_nudge = {}  # populated when nudge is requested
 
        need_nexus_sums = bool(self._nex_id_to_pos) and 'streamflow' in variables
        need_nudge_sums = bool(self._nex_id_to_pos) and 'nudge' in variables
        
        # Loop through results list and save needed output values
        for r in run_results:
            seg_ids = r[0]
            fvd = r[1]
 
            if seg_ids is None or len(seg_ids) == 0 or fvd is None:
                continue
            
            seg_ids = np.asarray(seg_ids, dtype=np.int64)
            wb_pos = np.array([self._wb_id_to_pos.get(int(s), -1)
                               for s in seg_ids], dtype=np.intp)
            wb_mask = wb_pos >= 0
 
            if wb_mask.any():
                valid_pos = wb_pos[wb_mask]
                for var in variables:
                    if var in _FVD_POSITIONS:
                        col = base_col + _FVD_POSITIONS[var]
                        output_dict[var][valid_pos] = fvd[wb_mask, col]
 
            # Save nexus upstream flows for aggregation
            if need_nexus_sums:
                flow_col = base_col + _FVD_POSITIONS['streamflow']
                flow_vals = fvd[:, flow_col]
                for sid, val in zip(seg_ids, flow_vals):
                    sid = int(sid)
                    if sid in self._nex_upstream_ids:
                        seg_flow[int(sid)] = float(val)
 
            if need_nudge_sums or ('nudge' in variables and self._wb_id_to_pos):
                self._collect_nudge(r, step_index, seg_nudge)
 
        # Write nudge to wb rows
        if 'nudge' in variables:
            #TODO: Validate that this works...
            for sid, val in seg_nudge.items():
                pos = self._wb_id_to_pos.get(sid)
                if pos is not None:
                    output_dict['nudge'][pos] = val
        
        # Aggregate for nexus rows
        for nid, pos in self._nex_id_to_pos.items():
            upstream = self._nex_upstream.get(nid, [])
            if not upstream:
                continue
 
            if 'streamflow' in variables:
                vals = [seg_flow[u] for u in upstream if u in seg_flow]
                if vals:
                    output_dict['streamflow'][pos] = sum(vals)
 
            if 'nudge' in variables:
                vals = [seg_nudge[u] for u in upstream if u in seg_nudge]
                if vals:
                    output_dict['nudge'][pos] = sum(vals)
        
        # Write to file
        for var, data in output_dict.items():
            self._nc.variables[var][nc_time_idx, :] = data
 
    def collect_nudge(self, r: Tuple, step_index: int,
                       seg_nudge: Dict[int, float]):
        """
        Pull nudge values for this timestep from r[8] into seg_nudge.
 
        r[8] is expected to be a pandas DataFrame indexed by segment ID
        with one column per internal timestep.  Only gaged reaches appear;
        un-gaged ones are simply absent from the DataFrame.
        """
        if len(r) <= 8:
            return
 
        nudge_df = r[8]
        if nudge_df is None or not isinstance(nudge_df, pd.DataFrame):
            return
        if nudge_df.empty or step_index >= len(nudge_df.columns):
            return
 
        col = nudge_df.iloc[:, step_index]
        for sid, val in col.items():
            if not np.isnan(val):
                seg_nudge[int(sid)] = float(val)