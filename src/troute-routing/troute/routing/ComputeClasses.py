import pandas as pd
import numpy as np
import pathlib
import xarray as xr
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from joblib import delayed, Parallel
import glob
import re
import time
import logging
from itertools import chain

LOG = logging.getLogger('')

# -----------------------------------------------------------------------------
# Abstract Compute Class:
#   Define all slots and pass function definitions to child classes
# -----------------------------------------------------------------------------
class AbstractCompute(ABC):
    """
    
    """
    __slots__ = ["_preprocessed_data", "_master_seg_index", "_master_lake_index",
                 "_preprocessed_data_old"]
    
    def __init__(self, network):
        self._preprocessed_data = {}
        self._preprocessed_data_old = {}
        self._master_seg_index = network.dataframe.index
        self._master_lake_index = (network.waterbody_dataframe.index 
                                   if not network.waterbody_dataframe.empty 
                                   else pd.Index([]))
    
    @abstractmethod
    def subset_network(self, network, data_assimilation):
        pass
    
    # @abstractmethod
    # def subset_forcing(self, network, data_assimilation):
    #     pass
    
    @abstractmethod
    def run_routing(self,):
        pass

# -----------------------------------------------------------------------------
# Base compute class definitions:
#   1. Serial: perform routing in serial, no parallelization
#   2. ByNetwork: parallelize by independent networks
#   3. BySubNetworkJIT: parallelize by independent sub-networks
#   4. BySubNetworkJITClustered: parallelize by clustered independent sub-networks
# -----------------------------------------------------------------------------

class Serial(AbstractCompute):
    """
    
    """
    def __init__(self, network):
        super().__init__(network)  
    
    def subset_network(self, network, dt):
        print("Starting subset pre-computation...")
        
        # 1. PRE-COMPUTE MASTER DATA OUTSIDE THE LOOP
        param_df = network.dataframe.copy()
        param_df["dt"] = dt
        param_df = param_df.astype("float32")
        
        # Isolate just the columns we need to slice later
        param_cols_to_keep = ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"]
        master_param_vals = param_df[param_cols_to_keep].values
        master_param_cols = np.array(param_cols_to_keep)
        
        # Convert heavy Pandas indexes to ultra-fast native Python sets
        master_seg_set = set(self._master_seg_index)
        
        has_waterbodies = not network.waterbody_dataframe.empty
        wb_cols = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE", "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]
        if has_waterbodies:
            master_lake_set = set(self._master_lake_index)
            master_wb_vals = network.waterbody_dataframe[wb_cols].values
            
            has_wb_types = not network.waterbody_types_dataframe.empty
            if has_wb_types:
                master_wb_types_vals = network.waterbody_types_dataframe["reservoir_type"].values
        else:
            master_lake_set = set()

        print("Starting optimized loop...")
        for twi, (tw, reach_list) in enumerate(network.reaches_by_tailwater.items(), 1):

            # 2. FAST TOPOLOGY SUBSETTING (Native Python)
            segs = list(chain.from_iterable(reach_list))
            segs_set = set(segs)
            
            # Native set operations are orders of magnitude faster
            common_segs_set = master_seg_set.intersection(segs_set)
            common_segs = sorted(list(common_segs_set)) # Sort ONCE here natively
            
            wbodies_segs = segs_set.symmetric_difference(common_segs_set)

            # 3. GET MASTER INDEXERS
            seg_indexer = self._master_seg_index.get_indexer(common_segs)
            
            # 4. INSTANT NUMPY SLICING (Bypassing .loc)
            param_vals_sub = master_param_vals[seg_indexer]
            param_idx_sub = np.array(common_segs, dtype=np.int64)

            # Waterbody subsetting
            if has_waterbodies:
                lake_segs_set = master_lake_set.intersection(segs_set)
                lake_segs = sorted(list(lake_segs_set))
                lake_indexer = self._master_lake_index.get_indexer(lake_segs)
                
                waterbodies_vals_sub = master_wb_vals[lake_indexer]
                
                if has_wb_types:
                    waterbody_types_vals_sub = master_wb_types_vals[lake_indexer].astype("int32")
                else:
                    waterbody_types_vals_sub = np.empty((0,), dtype="int32")
            else:
                lake_segs = []
                lake_indexer = np.array([], dtype=np.intp)
                waterbodies_vals_sub = np.empty((0, len(wb_cols)))
                waterbody_types_vals_sub = np.empty((0,), dtype="int32")

            reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

            # 5. REPLACING .REINDEX() WITH NUMPY VSTACK
            # Instead of a heavy pandas reindex, we create the NaN padding for lakes 
            # and stack it directly onto our numpy arrays, then sort the arrays based on the combined index.
            if lake_segs:
                combined_idx = np.concatenate([param_idx_sub, np.array(lake_segs, dtype=np.int64)])
                sort_order = np.argsort(combined_idx)
                
                final_param_idx = combined_idx[sort_order]
                
                lake_padding = np.full((len(lake_segs), master_param_vals.shape[1]), np.nan, dtype=np.float32)
                combined_vals = np.vstack([param_vals_sub, lake_padding])
                final_param_vals = combined_vals[sort_order]
            else:
                final_param_idx = param_idx_sub
                final_param_vals = param_vals_sub

            # 6. PACKING DICTIONARY
            self._preprocessed_data[tw] = {
                "reach_list": reach_list,
                "reaches_list_with_type": reaches_list_with_type,
                "upstream_connections": network.independent_networks[tw],
                
                "param_idx": final_param_idx,
                "param_cols": master_param_cols,
                "param_vals": final_param_vals,
                
                "lake_segs": lake_segs,
                "waterbodies_vals": waterbodies_vals_sub,
                "waterbody_types_vals": waterbody_types_vals_sub,
                
                "seg_indexer": seg_indexer,
                "lake_indexer": lake_indexer,
            }
    
    def subset_network_old(self, network, dt):
        print("Starting subset...")
        param_df = network.dataframe
        param_df["dt"] = dt
        param_df = param_df.astype("float32")
        
        print("Starting loop...")
        for twi, (tw, reach_list) in enumerate(network.reaches_by_tailwater.items(), 1):

            # Topology subsetting
            segs = list(chain.from_iterable(reach_list))
            common_segs = param_df.index.intersection(segs)
            wbodies_segs = set(segs).symmetric_difference(common_segs)

            # Get indices for segments (for qlat & q0)
            seg_indexer = self._master_seg_index.get_indexer(common_segs)
            
            # Waterbody subsetting
            waterbody_types_df_sub = pd.DataFrame()

            if not network.waterbody_dataframe.empty:
                lake_segs = list(network.waterbody_dataframe.index.intersection(segs))
                lake_indexer = self._master_lake_index.get_indexer(lake_segs)
                
                waterbodies_df_sub = network.waterbody_dataframe.loc[
                    lake_segs, ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE", "WeirC", 
                                "WeirE", "WeirL", "ifd", "qd0", "h0"],
                ]
                if not network.waterbody_types_dataframe.empty:
                    waterbody_types_df_sub = network.waterbody_types_dataframe.loc[lake_segs, ["reservoir_type"]]
            else:
                lake_segs = []
                lake_indexer = np.array([], dtype=np.intp)
                waterbodies_df_sub = pd.DataFrame()

            # Parameter subsetting
            param_df_sub = param_df.loc[
                common_segs, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
            ].sort_index()

            reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

            param_df_sub = param_df_sub.reindex(
                param_df_sub.index.tolist() + lake_segs
            ).sort_index()
            
            self._preprocessed_data_old[tw] = {
                # Top-level args
                # "qts_subdivisions": qts_subdivisions, 
                "reach_list": reach_list,
                "reaches_list_with_type": reaches_list_with_type,
                "upstream_connections": network.independent_networks[tw],
                
                # Parameters & Flows
                "param_idx": param_df_sub.index.values.astype("int64"),
                "param_cols": param_df_sub.columns.values,
                "param_vals": param_df_sub.values,
                
                # Lakes
                "lake_segs": lake_segs,
                "waterbodies_vals": waterbodies_df_sub.values,
                "waterbody_types_vals": waterbody_types_df_sub.values.astype("int32"),
                
                # Dynamic Indices (integer locators for slicing later)
                "seg_indexer": seg_indexer,
                "lake_indexer": lake_indexer,
                # "usgs_indexer": usgs_indexer,
                # "rfc_indexer": rfc_indexer,
            }
    
    def run_routing(
        self,
        compute_func,
        nts,
        dt,
        qts_subdivisions,
        q0_array,             # Expected as a 1D raw numpy array of the global q0
        qlat_array,           # Expected as a 1D raw numpy array of the global qlat
        usgs_df,              # Global DA DataFrames passed in for the unmodified helpers
        lastobs_df,
        reservoir_usgs_df,
        reservoir_usgs_param_df,
        reservoir_usace_df,
        reservoir_usace_param_df,
        reservoir_rfc_df,
        reservoir_rfc_param_df,
        great_lakes_df,
        great_lakes_param_df,
        great_lakes_climatology_df,
        data_assimilation_parameters,
        waterbody_type_specified,
        t0,
        da_decay_coefficient,
        assume_short_ts,
        return_courant,
        from_files,
    ):
        """
        Executes the compute_func over all sub-networks using the preprocessed data.
        """
        results = []

        for tw, pre_data in self._preprocessed_data.items():
            
            # ---------------------------------------------------------
            # 1. Unpack Local Pointers (Zero memory overhead, instant access)
            # ---------------------------------------------------------
            reach_list = pre_data["reach_list"]
            reaches_list_with_type = pre_data["reaches_list_with_type"]
            upstream_connections = pre_data["upstream_connections"]
            
            param_idx = pre_data["param_idx"]
            param_cols = pre_data["param_cols"]
            param_vals = pre_data["param_vals"]
            
            lake_segs = pre_data["lake_segs"]
            waterbodies_vals = pre_data["waterbodies_vals"]
            waterbody_types_vals = pre_data["waterbody_types_vals"]
            seg_indexer = pre_data["seg_indexer"]

            # ---------------------------------------------------------
            # 2. Fast Slicing of Dynamic Forcing Data (q0, qlat)
            # ---------------------------------------------------------
            # Slice the global arrays using our precomputed integer indexer
            q0_sub_vals = q0_array[seg_indexer].astype("float32")
            qlat_sub_vals = qlat_array[seg_indexer].astype("float32")
            
            # The original compute code expects q0 and qlat to be padded with NaNs 
            # for the lake segments and sorted to match param_idx. We can do this 
            # very quickly in numpy:
            if lake_segs:
                # Get common_segs back by slicing param_idx (it contains both)
                common_segs = [seg for seg in param_idx if seg not in lake_segs]
                combined_idx = np.concatenate([common_segs, lake_segs])
                sort_order = np.argsort(combined_idx)
                
                lake_pad = np.full(len(lake_segs), np.nan, dtype=np.float32)
                q0_sub_vals = np.concatenate([q0_sub_vals, lake_pad])[sort_order]
                qlat_sub_vals = np.concatenate([qlat_sub_vals, lake_pad])[sort_order]

            # ---------------------------------------------------------
            # 3. Call Unmodified DA Helper Functions
            # ---------------------------------------------------------
            # We recreate a tiny pandas index for the common segments to feed the helpers
            common_segs_idx = pd.Index([seg for seg in param_idx if seg not in lake_segs])
            
            usgs_df_sub, lastobs_df_sub, da_pos_byseg = _prep_da_dataframes(
                usgs_df, lastobs_df, common_segs_idx
            )
            da_pos_byreach, da_pos_bygage = _prep_da_positions_byreach(
                reach_list, lastobs_df_sub.index
            )

            # Re-wrap waterbody types briefly to satisfy the unmodified helper
            wb_types_df_sub = pd.DataFrame(
                waterbody_types_vals, index=lake_segs, columns=["reservoir_type"]
            ) if lake_segs else pd.DataFrame()

            (
                res_usgs_df_sub, res_usgs_df_time, res_usgs_update_time, 
                res_usgs_prev_flow, res_usgs_pers_update, res_usgs_pers_idx,
                res_usace_df_sub, res_usace_df_time, res_usace_update_time, 
                res_usace_prev_flow, res_usace_pers_update, res_usace_pers_idx,
                res_rfc_df_sub, res_rfc_totalCounts, res_rfc_file, 
                res_rfc_use_forecast, res_rfc_ts_idx, res_rfc_update_time, 
                res_rfc_da_timestep, res_rfc_persist_days,
                gl_df_sub, gl_parm_lake_id_sub, gl_param_flows_sub, 
                gl_param_time_sub, gl_param_update_time_sub, 
                gl_climatology_df_sub, wb_types_df_sub_out
            ) = _prep_reservoir_da_dataframes(
                reservoir_usgs_df, reservoir_usgs_param_df,
                reservoir_usace_df, reservoir_usace_param_df,
                reservoir_rfc_df, reservoir_rfc_param_df,
                great_lakes_df, great_lakes_param_df, great_lakes_climatology_df,
                wb_types_df_sub, t0, from_files,
            )

            # ---------------------------------------------------------
            # 4. Execute the Compute Function
            # ---------------------------------------------------------
            results.append(
                compute_func(
                    nts,
                    dt,
                    qts_subdivisions,
                    reaches_list_with_type,
                    upstream_connections,
                    param_idx,
                    param_cols,
                    param_vals,
                    q0_sub_vals,
                    qlat_sub_vals,
                    lake_segs,
                    waterbodies_vals,
                    data_assimilation_parameters,
                    waterbody_types_vals,
                    waterbody_type_specified,
                    t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    
                    # Core USGS DA
                    usgs_df_sub.values.astype("float32"),
                    np.array(da_pos_byseg, dtype="int32"),
                    np.array(da_pos_byreach, dtype="int32"),
                    np.array(da_pos_bygage, dtype="int32"),
                    lastobs_df_sub.get("lastobs_discharge", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    lastobs_df_sub.get("time_since_lastobs", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    da_decay_coefficient,
                    
                    # USGS Hybrid Reservoir DA data
                    res_usgs_df_sub.values.astype("float32"),
                    res_usgs_df_sub.index.values.astype("int32"),
                    res_usgs_df_time.astype('float32'),
                    res_usgs_update_time.astype('float32'),
                    res_usgs_prev_flow.astype('float32'),
                    res_usgs_pers_update.astype('float32'),
                    res_usgs_pers_idx.astype('float32'),
                    
                    # USACE Hybrid Reservoir DA data
                    res_usace_df_sub.values.astype("float32"),
                    res_usace_df_sub.index.values.astype("int32"),
                    res_usace_df_time.astype('float32'),
                    res_usace_update_time.astype("float32"),
                    res_usace_prev_flow.astype("float32"),
                    res_usace_pers_update.astype("float32"),
                    res_usace_pers_idx.astype("float32"),
                    
                    # RFC Reservoir DA data
                    res_rfc_df_sub.values.astype("float32"),
                    res_rfc_df_sub.index.values.astype("int32"),
                    res_rfc_totalCounts.astype("int32"),
                    res_rfc_file,
                    res_rfc_use_forecast.astype("int32"),
                    res_rfc_ts_idx.astype("int32"),
                    res_rfc_update_time.astype("float32"),
                    res_rfc_da_timestep.astype("int32"),
                    res_rfc_persist_days.astype("int32"),
                    
                    # Great Lakes DA data
                    gl_df_sub.lake_id.values.astype("int32"),
                    gl_df_sub.time.values.astype("int32"),
                    gl_df_sub.Discharge.values.astype("float32"),
                    gl_parm_lake_id_sub.astype("int32"),
                    gl_param_flows_sub.astype("float32"),
                    gl_param_time_sub.astype("int32"),
                    gl_param_update_time_sub.astype("int32"),
                    gl_climatology_df_sub.values.astype("float32"),
                    
                    # Tail args
                    {},
                    assume_short_ts,
                    return_courant,
                    from_files=from_files,
                )
            )

        return results
            
    # def run_routing(self, network, usgs_df, lastobs_df, compute_func):
        
    #     results = []
    #     for tw, pre_data in self._preprocessed_data.items():
            
    #         param_df_index = pre_data['seg_indexer']
    #         qlat_sub = network._qlateral[param_df_index]
    #         q0_sub = network.q0[param_df_index]
            
    #         usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_index)
    #         da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(pre_data['reach_list'], lastobs_df_sub.index)

    #         qlat_sub = qlat_sub.reindex(param_df_index)
    #         q0_sub = q0_sub.reindex(param_df_index)


class ByNetwork(AbstractCompute):
    pass


class BySubNetworkJIT(AbstractCompute):
    pass


class BySubNetworkJITClustered(AbstractCompute):
    pass


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _build_reach_type_list(reach_list, wbodies_segs):
    reach_type_list_array = np.zeros(len(reach_list), dtype=np.uint8)
    for i in range(len(reach_list)):
        reaches = reach_list[i]
        for reach in reaches:
            if reach in wbodies_segs:
                reach_type_list_array[i] = 1
                break
    
    return list(zip(reach_list, reach_type_list_array))

def _prep_da_dataframes(
    usgs_df,
    lastobs_df,
    param_df_sub_idx,
    exclude_segments=None,
    ):
    """
    Produce, based on the segments in the param_df_sub_idx (which is a subset
    representing a subnetwork of the larger collection of all segments),
    a subset of the relevant usgs gage observation time series
    and the relevant last-valid gage observation from any
    prior model execution.
    
    exclude_segments (list): segments to exclude from param_df_sub when searching for gages
                             This catches and excludes offnetwork upstreams segments from being
                             realized as locations for DA substitution. Else, by-subnetwork
                             parallel executions fail.

    Cases to consider:
    USGS_DF, LAST_OBS
    Yes, Yes: Analysis and Assimilation; Last_Obs used to fill gaps in the front of the time series
    No, Yes: Forecasting mode;
    Yes, No; Cold-start case;
    No, No: Open-Loop;

    For both cases where USGS_DF is present, there is a sub-case where the length of the observed
    time series is as long as the simulation.

    """
    
    subnet_segs = param_df_sub_idx
    # segments in the subnetwork ONLY, no offnetwork upstreams included
    if exclude_segments:
        subnet_segs = param_df_sub_idx.difference(set(exclude_segments))
    
    # NOTE: Uncomment to easily test no observations...
    # usgs_df = pd.DataFrame()
    if not usgs_df.empty and not lastobs_df.empty:
        # index values for last obs are not correct, but line up correctly with usgs values. Switched
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        usgs_segs = (usgs_df.index.
                     intersection(subnet_segs).
                     reindex(lastobs_segs)[0].
                     to_list()
                    )
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
    elif usgs_df.empty and not lastobs_df.empty:
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        # Create a completely empty list of gages -- the .shape[1] attribute
        # will be == 0, and that will trigger a reference to the lastobs.
        # in the compute kernel below.
        usgs_df_sub = pd.DataFrame(index=lastobs_df_sub.index,columns=[])
        usgs_segs = lastobs_segs
        da_positions_list_byseg = param_df_sub_idx.get_indexer(lastobs_segs)
    elif not usgs_df.empty and lastobs_df.empty:
        usgs_segs = list(usgs_df.index.intersection(subnet_segs))
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
        lastobs_df_sub = pd.DataFrame(index=usgs_df_sub.index,columns=["discharge","time","model_discharge"])
    else:
        usgs_df_sub = pd.DataFrame()
        lastobs_df_sub = pd.DataFrame()
        da_positions_list_byseg = []

    return usgs_df_sub, lastobs_df_sub, da_positions_list_byseg


def _prep_da_positions_byreach(reach_list, gage_index):
    """
    produce a list of indexes of the reach_list identifying reaches with gages
    and a corresponding list of indexes of the gage_list of the gages in
    the order they are found in the reach_list.
    """
    reach_key = []
    reach_gage = []
    for i, r in enumerate(reach_list):
        for s in r:
            if s in gage_index:
                reach_key.append(i)
                reach_gage.append(s)
    gage_reach_i = gage_index.get_indexer(reach_gage)

    return reach_key, gage_reach_i

def _prep_reservoir_da_dataframes(reservoir_usgs_df,
                                  reservoir_usgs_param_df,
                                  reservoir_usace_df,
                                  reservoir_usace_param_df,
                                  reservoir_rfc_df,
                                  reservoir_rfc_param_df,
                                  great_lakes_df,
                                  great_lakes_param_df,
                                  great_lakes_climatology_df,
                                  waterbody_types_df_sub,
                                  t0, 
                                  from_files,
                                  exclude_segments=None):
    '''
    Helper function to build reservoir DA data arrays for routing computations

    Arguments
    ---------
    reservoir_usgs_df        (DataFrame): gage flow observations at USGS-type reservoirs
    reservoir_usgs_param_df  (DataFrame): USGS reservoir DA state parameters
    reservoir_usace_df       (DataFrame): gage flow observations at USACE-type reservoirs
    reservoir_usace_param_df (DataFrame): USACE reservoir DA state parameters
    reservoir_rfc_df         (DataFrame): gage flow observations and forecasts at RFC-type reservoirs
    reservoir_rfc_param_df   (DataFrame): RFC reservoir DA state parameters
    waterbody_types_df_sub   (DataFrame): type-codes for waterbodies in sub domain
    t0                        (datetime): model initialization time

    Returns
    -------
    * there are many returns, because we are passing explicit arrays to mc_reach cython code
    reservoir_usgs_df_sub                 (DataFrame): gage flow observations for USGS-type reservoirs in sub domain
    reservoir_usgs_df_time                  (ndarray): time in seconds from model initialization time
    reservoir_usgs_update_time              (ndarray): update time (sec) to search for new observation at USGS reservoirs
    reservoir_usgs_prev_persisted_flow      (ndarray): previously persisted outflow rates at USGS reservoirs
    reservoir_usgs_persistence_update_time  (ndarray): update time (sec) of persisted value at USGS reservoirs
    reservoir_usgs_persistence_index        (ndarray): index denoting elapsed persistence epochs at USGS reservoirs
    reservoir_usace_df_sub                (DataFrame): gage flow observations for USACE-type reservoirs in sub domain
    reservoir_usace_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usace_update_time             (ndarray): update time (sec) to search for new observation at USACE reservoirs
    reservoir_usace_prev_persisted_flow     (ndarray): previously persisted outflow rates at USACE reservoirs
    reservoir_usace_persistence_update_time (ndarray): update time (sec) of persisted value at USACE reservoirs
    reservoir_usace_persistence_index       (ndarray): index denoting elapsed persistence epochs at USACE reservoirs

    '''
    if not reservoir_usgs_df.empty:
        usgs_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==2
                                ].index
        if exclude_segments:
            usgs_wbodies_sub = list(set(usgs_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usgs_df_sub = reservoir_usgs_df.loc[usgs_wbodies_sub]
        reservoir_usgs_df_time = []
        for timestamp in reservoir_usgs_df.columns:
            reservoir_usgs_df_time.append((timestamp - t0).total_seconds())
        reservoir_usgs_df_time = np.array(reservoir_usgs_df_time)
        reservoir_usgs_update_time = reservoir_usgs_param_df['update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_prev_persisted_flow = reservoir_usgs_param_df['prev_persisted_outflow'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_update_time = reservoir_usgs_param_df['persistence_update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_index = reservoir_usgs_param_df['persistence_index'].loc[usgs_wbodies_sub].to_numpy()
    else:
        reservoir_usgs_df_sub = pd.DataFrame()
        reservoir_usgs_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 2] = 1

    # select USACE reservoir DA data waterbodies in sub-domain
    if not reservoir_usace_df.empty:
        usace_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==3
                                ].index
        if exclude_segments:
            usace_wbodies_sub = list(set(usace_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usace_df_sub = reservoir_usace_df.loc[usace_wbodies_sub]
        reservoir_usace_df_time = []
        for timestamp in reservoir_usace_df.columns:
            reservoir_usace_df_time.append((timestamp - t0).total_seconds())
        reservoir_usace_df_time = np.array(reservoir_usace_df_time)
        reservoir_usace_update_time = reservoir_usace_param_df['update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_prev_persisted_flow = reservoir_usace_param_df['prev_persisted_outflow'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_update_time = reservoir_usace_param_df['persistence_update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_index = reservoir_usace_param_df['persistence_index'].loc[usace_wbodies_sub].to_numpy()
    else: 
        reservoir_usace_df_sub = pd.DataFrame()
        reservoir_usace_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 3] = 1
    
    # RFC reservoirs
    if not reservoir_rfc_df.empty:
        rfc_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==4
            ].index
        if exclude_segments:
            rfc_wbodies_sub = list(set(rfc_wbodies_sub).difference(set(exclude_segments)))
        reservoir_rfc_df_sub = reservoir_rfc_df.loc[rfc_wbodies_sub]
        reservoir_rfc_totalCounts = reservoir_rfc_param_df['totalCounts'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_file = reservoir_rfc_param_df['file'].loc[rfc_wbodies_sub].to_list()
        reservoir_rfc_use_forecast = reservoir_rfc_param_df['use_rfc'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_timeseries_idx = reservoir_rfc_param_df['timeseries_idx'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_update_time = reservoir_rfc_param_df['update_time'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_da_timestep = reservoir_rfc_param_df['da_timestep'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_persist_days = reservoir_rfc_param_df['rfc_persist_days'].loc[rfc_wbodies_sub].to_numpy()
    else:
        reservoir_rfc_df_sub = pd.DataFrame()
        reservoir_rfc_totalCounts = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_file = []
        reservoir_rfc_use_forecast = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_timeseries_idx = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_da_timestep = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_persist_days = pd.DataFrame().to_numpy().reshape(0,)
        if not from_files:
            if not waterbody_types_df_sub.empty:
                waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 4] = 1
    
    # Great Lakes
    if not great_lakes_df.empty:
        gl_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==6
            ].index
        if exclude_segments:
            gl_wbodies_sub = list(set(gl_wbodies_sub).difference(set(exclude_segments)))
        gl_df_sub = great_lakes_df[great_lakes_df['lake_id'].isin(gl_wbodies_sub)]
        gl_climatology_df_sub = great_lakes_climatology_df.loc[gl_wbodies_sub]
        gl_param_df_sub = great_lakes_param_df[great_lakes_param_df['lake_id'].isin(gl_wbodies_sub)]
        gl_parm_lake_id_sub = gl_param_df_sub.lake_id.to_numpy()
        gl_param_flows_sub = gl_param_df_sub.previous_assimilated_outflows.to_numpy()
        gl_param_time_sub = gl_param_df_sub.previous_assimilated_time.to_numpy()
        gl_param_update_time_sub = gl_param_df_sub.update_time.to_numpy()
    else:
        gl_df_sub = pd.DataFrame(columns=['lake_id','time','Discharge'])
        gl_climatology_df_sub = pd.DataFrame()
        gl_parm_lake_id_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_flows_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_time_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_update_time_sub = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 6] = 1

    return (
        reservoir_usgs_df_sub, reservoir_usgs_df_time, reservoir_usgs_update_time, reservoir_usgs_prev_persisted_flow, reservoir_usgs_persistence_update_time, reservoir_usgs_persistence_index,
        reservoir_usace_df_sub, reservoir_usace_df_time, reservoir_usace_update_time, reservoir_usace_prev_persisted_flow, reservoir_usace_persistence_update_time, reservoir_usace_persistence_index,
        reservoir_rfc_df_sub, reservoir_rfc_totalCounts, reservoir_rfc_file, reservoir_rfc_use_forecast, reservoir_rfc_timeseries_idx, reservoir_rfc_update_time, reservoir_rfc_da_timestep, reservoir_rfc_persist_days,
        gl_df_sub, gl_parm_lake_id_sub, gl_param_flows_sub, gl_param_time_sub, gl_param_update_time_sub, gl_climatology_df_sub,
        waterbody_types_df_sub
        )