import argparse
import time
import math
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd

from .input import _input_handler_v03
from .preprocess import (
    nwm_network_preprocess,
    nwm_initial_warmstate_preprocess,
    nwm_forcing_preprocess,
    unpack_nwm_preprocess_data,
)
from .output import nwm_output_generator
from .log_level_set import log_level_set
from troute.routing.compute import compute_nhd_routing_v02, compute_diffusive_routing
import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io
import troute.nhd_network_utilities_v02 as nnu
import troute.routing.diffusive_utils as diff_utils

LOG = logging.getLogger('')

def _handle_args_v03(argv):
    '''
    Handle command line input argument - filepath of configuration file
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--custom-input-file",
        dest="custom_input_file",
        help="Path of a .yaml or .json file containing model configuration parameters. See doc/v3_doc.yaml",
    )
    return parser.parse_args(argv)

def nwm_route(
    downstream_connections,
    upstream_connections,
    waterbodies_in_connections,
    reaches_bytw,
    parallel_compute_method,
    compute_kernel,
    subnetwork_target_size,
    cpu_pool,
    t0,
    dt,
    nts,
    qts_subdivisions,
    independent_networks,
    param_df,
    q0,
    qlats,
    usgs_df,
    lastobs_df,
    reservoir_usgs_df,
    reservoir_usgs_param_df,
    reservoir_usace_df,
    reservoir_usace_param_df,
    da_parameter_dict,
    assume_short_ts,
    return_courant,
    waterbodies_df,
    waterbody_parameters,
    waterbody_types_df,
    waterbody_type_specified,
    diffusive_parameters,
    diffusive_network_data,
    topobathy_data,
    subnetwork_list,
):

    ################### Main Execution Loop across ordered networks
    
    
    start_time = time.time()

    if return_courant:
        LOG.info(
            f"executing routing computation, with Courant evaluation metrics returned"
        )
    else:
        LOG.info(f"executing routing computation ...")

    start_time_mc = time.time()
    results = compute_nhd_routing_v02(
        downstream_connections,
        upstream_connections,
        waterbodies_in_connections,
        reaches_bytw,
        compute_kernel,
        parallel_compute_method,
        subnetwork_target_size,  # The default here might be the whole network or some percentage...
        cpu_pool,
        t0,
        dt,
        nts,
        qts_subdivisions,
        independent_networks,
        param_df,
        q0,
        qlats,
        usgs_df,
        lastobs_df,
        reservoir_usgs_df,
        reservoir_usgs_param_df,
        reservoir_usace_df,
        reservoir_usace_param_df,
        da_parameter_dict,
        assume_short_ts,
        return_courant,
        waterbodies_df,
        waterbody_parameters,
        waterbody_types_df,
        waterbody_type_specified,
        subnetwork_list,
    )
    
    # returns list, first item is run result, second item is subnetwork items
    subnetwork_list = results[1]
    results = results[0]
    
    if diffusive_network_data: # run diffusive side of a hybrid simulation
        
        LOG.debug("MC computation complete in %s seconds." % (time.time() - start_time_mc))
        start_time_diff = time.time()
                
        # call diffusive wave simulation and append results to MC results
        results.extend(
            compute_diffusive_routing(
                results,
                diffusive_network_data,
                cpu_pool,
                dt,
                nts,
                q0,
                qlats,
                qts_subdivisions,
                usgs_df,
                lastobs_df,
                da_parameter_dict,
                diffusive_parameters,
                waterbodies_df,
                topobathy_data,
            )
        )
        LOG.debug("Diffusive computation complete in %s seconds." % (time.time() - start_time_diff))

    LOG.debug("ordered reach computation complete in %s seconds." % (time.time() - start_time))

    return results, subnetwork_list


def new_nwm_q0(run_results):
    """
    Prepare a new q0 dataframe with initial flow and depth to act as
    a warmstate for the next simulation chunk.
    """
    return pd.concat(
        # TODO: we only need two fields, technically, and the restart file produced by WRF-Hydro
        # actually contains a field qu0, which is never used for restart (the qu0 can be obtained
        # as the qd0 from the topologically upstream segments, just like during the calculation).
        # In any case, the qu0 currently in the WRF-Hydro output is populated with the same value
        # as the qd0.
        # [pd.DataFrame(d[:,-3::2], index=i, columns=["qd0", "h0"]) for i, d in run_results],
        # [pd.DataFrame(r[1][:,-3:], index=r[0], columns=["qu0", "v0", "h0"]) for r in run_results],
        [
            pd.DataFrame(
                r[1][:, [-3, -3, -1]], index=r[0], columns=["qu0", "qd0", "h0"]
            )
            for r in run_results
        ],
        copy=False,
    )


def create_lite_restart_time_index(run_sets, time_interval):
    """
    Create an index of datetimes from specified lite restart time intervals.
    This will filter the larger dataframe that gets passed to create_lite_restart_df
    function.
    """
    if len(run_sets) > 1:
        run_sets_count = len(run_sets)
        initial_timestamp = run_sets[0].get('t0')
        final_timestamp = run_sets[run_sets_count - 1].get('final_timestamp')
    
    else:
        initial_timestamp = run_sets[0].get('t0')
        final_timestamp = run_sets[0].get('final_timestamp')
    
    return pd.date_range(initial_timestamp,
                         final_timestamp,
                         freq=str(time_interval) + 'S')
    
    
def create_lite_restart_df(run_results, run_set, lite_restart_idx):
    """
    Prepare a q0 dataframe with flow and depth at specified time intervals.
    """

    time_steps = []
    i = 0
    ind = []
    for number in range(run_results[0][1].shape[1]//3):
        time_steps.append('_' + str(number))
        ind.append([i,i,i+2])
        i += 3
    ind = [item for sublist in ind for item in sublist]
    column_names = [x + str(y) for y in time_steps for x in ["qu0", "qd0", "h0"]]
    df = pd.concat(
        [
            pd.DataFrame(
                r[1][:,ind], index=r[0], columns=column_names
            )
            for r in run_results
        ],
        copy=False,
    )
    
    t0 = run_set.get('t0')
    dt = run_set.get('dt')
    final_t = run_set.get('final_timestamp')
    full_timesteps = pd.date_range(t0, final_t, freq=str(dt) + 'S', closed='right')
    full_timesteps = np.repeat(full_timesteps,3)
    
    column_ind = full_timesteps.isin(lite_restart_idx)
    df = df.iloc[:,column_ind]
    
    dfList = []
    for i in range(0, df.shape[1], 3):
        dfList.append(df.iloc[:, i:i+3])
    
    q0_dict = dict(zip(full_timesteps[column_ind].unique(), dfList))
    
    return q0_dict


def get_waterbody_water_elevation(waterbodies_df, q0):
    """
    Update the starting water_elevation of each lake/reservoir
    with depth values from q0
    """
    waterbodies_df.update(q0)

    return waterbodies_df

def set_reservoir_da_prams(run_results):
    '''
    Update persistence reservoir DA parameters for subsequent loops
    '''
    
    reservoir_usgs_param_df = pd.DataFrame(data = [], 
                                           index = [], 
                                           columns = [
                                               'update_time', 'prev_persisted_outflow', 
                                               'persistence_update_time', 'persistence_index'
                                           ]
                                          )
    reservoir_usace_param_df = pd.DataFrame(data = [], 
                                           index = [], 
                                           columns = [
                                               'update_time', 'prev_persisted_outflow', 
                                               'persistence_update_time', 'persistence_index'
                                           ]
                                          )
    
    for r in run_results:
        
        if len(r[4][0]) > 0:
            tmp_usgs = pd.DataFrame(data = r[4][1], index = r[4][0], columns = ['update_time'])
            tmp_usgs['prev_persisted_outflow'] = r[4][2]
            tmp_usgs['persistence_update_time'] = r[4][4]
            tmp_usgs['persistence_index'] = r[4][3]
            reservoir_usgs_param_df = pd.concat([reservoir_usgs_param_df, tmp_usgs])
        
        if len(r[5][0]) > 0:
            tmp_usace = pd.DataFrame(data = r[5][1], index = r[5][0], columns = ['update_time'])
            tmp_usace['prev_persisted_outflow'] = r[5][2]
            tmp_usace['persistence_update_time'] = r[5][4]
            tmp_usace['persistence_index'] = r[5][3]
            reservoir_usace_param_df = pd.concat([reservoir_usace_param_df, tmp_usace])
    
    return reservoir_usgs_param_df, reservoir_usace_param_df


def update_lookback_hours(dt, nts, waterbody_parameters):
    """
    Update the lookback hours that an RFC type reservoir searches in reverse
    from the model start time to find a time series file. The update is based
    on the total hours ran in the prior loop.
    """

    waterbody_parameters['rfc']['reservoir_rfc_forecasts_lookback_hours'] = \
    waterbody_parameters['rfc']['reservoir_rfc_forecasts_lookback_hours'] + \
    math.ceil((dt * nts) / 3600)

    return waterbody_parameters


def new_lastobs(run_results, time_increment):
    """
    Creates new "lastobs" dataframe for the next simulation chunk.

    run_results - output from the compute kernel sequence, organized
        (because that is how it comes out of the kernel) by network.
        For each item in the result, there are four elements, the
        fourth of which is a tuple containing: 1) a list of the
        segments ids where data assimilation was performed (if any)
        in that network; 2) a list of the last valid observation
        applied at that segment; 3) a list of the time in seconds
        from the beginning of the last simulation that the
        observation was applied.
    time_increment - length of the prior simulation. To prepare the
        next lastobs state, we have to convert the time since the prior
        simulation start to a time since the new simulation start.
        If the most recent observation was right at the end of the
        prior loop, then the value in the incoming run_result will
        be equal to the time_increment and the output value will be
        zero. If observations were not present at the last timestep,
        the last obs time will be calculated to a negative value --
        the number of seconds ago that the last valid observation
        was used for assimilation.
    """
    df = pd.concat(
        [
            pd.DataFrame(
                # TODO: Add time_increment (or subtract?) from time_since_lastobs
                np.array([rr[3][1],rr[3][2]]).T,
                index=rr[3][0],
                columns=["time_since_lastobs", "lastobs_discharge"]
            )
            for rr in run_results
            if not rr[3][0].size == 0
        ],
        copy=False,
    )
    df["time_since_lastobs"] = df["time_since_lastobs"] - time_increment

    return df


def main_v03(argv):
    """
    High level orchestration of t-route simulations for NWM application
    """
    args = _handle_args_v03(argv)
    
    # unpack user inputs
    (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        diffusive_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)
   
    showtiming = log_parameters.get("showtiming", None)
    if showtiming:
        task_times = {}
        task_times['initial_condition_time'] = 0
        task_times['forcing_time'] = 0
        task_times['route_time'] = 0
        task_times['output_time'] = 0
        main_start_time = time.time()

    if showtiming:
        network_start_time = time.time()

    # Build routing network data objects. Network data objects specify river 
    # network connectivity, channel geometry, and waterbody parameters.
    if preprocessing_parameters.get('use_preprocessed_data', False): 
        
        # get data from pre-processed file
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            link_lake_crosswalk,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
            usgs_lake_gage_crosswalk, 
            usace_lake_gage_crosswalk,
            diffusive_network_data,
            topobathy_data,
        ) = unpack_nwm_preprocess_data(
            preprocessing_parameters
        )
    else:
        
        # build data objects from scratch
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            link_lake_crosswalk,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
            usgs_lake_gage_crosswalk, 
            usace_lake_gage_crosswalk,
            diffusive_network_data,
            topobathy_data,
        ) = nwm_network_preprocess(
            supernetwork_parameters,
            waterbody_parameters,
            preprocessing_parameters,
            compute_parameters,
            data_assimilation_parameters,
        )
    
    if showtiming:
        network_end_time = time.time()
        task_times['network_time'] = network_end_time - network_start_time

    # list of all segments in the domain (MC + diffusive)
    segment_index = param_df.index
    if diffusive_network_data:
        for tw in diffusive_network_data:
            segment_index = segment_index.append(
                pd.Index(diffusive_network_data[tw]['mainstem_segs'])
            ) 
    
    # TODO: This function modifies one of its arguments (waterbodies_df), which is somewhat poor practice given its otherwise functional nature. Consider refactoring
    waterbodies_df, q0, t0, lastobs_df, da_parameter_dict = nwm_initial_warmstate_preprocess(
        break_network_at_waterbodies,
        restart_parameters,
        data_assimilation_parameters,
        segment_index,
        waterbodies_df,
        link_lake_crosswalk,
    )
    
    if showtiming:
        ic_end_time = time.time()
        task_times['initial_condition_time'] += ic_end_time - network_end_time

    # Create run_sets: sets of forcing files for each loop
    run_sets = nnu.build_forcing_sets(forcing_parameters, t0)

    # Create da_sets: sets of TimeSlice files for each loop
    if "data_assimilation_parameters" in compute_parameters:
        da_sets = nnu.build_da_sets(data_assimilation_parameters, run_sets, t0)
        
    # Create parity_sets: sets of CHRTOUT files against which to compare t-route flows
    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
    else:
        parity_sets = []
    
    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    cpu_pool = compute_parameters.get("cpu_pool", None)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)

    (
        qlats, 
        usgs_df, 
        reservoir_usgs_df, 
        reservoir_usgs_param_df,
        reservoir_usace_df,
        reservoir_usace_param_df
    ) = nwm_forcing_preprocess(
        run_sets[0],
        forcing_parameters,
        da_sets[0] if data_assimilation_parameters else {},
        data_assimilation_parameters,
        break_network_at_waterbodies,
        segment_index,
        link_gage_df,
        usgs_lake_gage_crosswalk, 
        usace_lake_gage_crosswalk,
        link_lake_crosswalk,
        lastobs_df.index,
        cpu_pool,
        t0,
    )
    
        
    if showtiming:
        forcing_end_time = time.time()
        task_times['forcing_time'] += forcing_end_time - ic_end_time

    # Create a time index of restart files based on user input
    if output_parameters["lite_restart"].get('lite_restart_output_time_interval'):
        lite_restart_idx = create_lite_restart_time_index(
            run_sets,
            output_parameters["lite_restart"].get("lite_restart_output_time_interval")
        )
    
    # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
    # on first iteration of for loop only. For additional loops this will be passed
    # to function from inital loop. 
    subnetwork_list = [None, None, None]
        
    for run_set_iterator, run in enumerate(run_sets):

        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        if showtiming:
            route_start_time = time.time()
        
        run_results = nwm_route(
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            cpu_pool,
            t0,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            usgs_df,
            lastobs_df,
            reservoir_usgs_df,
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_parameters,
            diffusive_network_data,
            topobathy_data,
            subnetwork_list,
        )
        
        # returns list, first item is run result, second item is subnetwork items
        subnetwork_list = run_results[1]
        run_results = run_results[0]
        
        if showtiming:
            route_end_time = time.time()
            task_times['route_time'] += route_end_time - route_start_time

        # create initial conditions for next loop itteration
        q0 = new_nwm_q0(run_results)
        waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)
        
        # get reservoir DA initial parameters for next loop itteration
        reservoir_usgs_param_df, reservoir_usace_param_df = set_reservoir_da_prams(run_results)
        
        # TODO move the conditional call to write_lite_restart to nwm_output_generator.
        if "lite_restart" in output_parameters:
            if output_parameters["lite_restart"].get('lite_restart_output_time_interval'):
                
                q0_multiple_timesteps = create_lite_restart_df(run_results, run, lite_restart_idx)
                
                nhd_io.write_lite_restart(
                q0, 
                waterbodies_df, 
                t0 + timedelta(seconds = dt * nts), 
                output_parameters['lite_restart'],
                q0_multiple_timesteps
            )
                
            else:
                
                nhd_io.write_lite_restart(
                    q0, 
                    waterbodies_df, 
                    t0 + timedelta(seconds = dt * nts), 
                    output_parameters['lite_restart']
                )
        
        if run_set_iterator < len(run_sets) - 1:
            (
                qlats, 
                usgs_df, 
                reservoir_usgs_df, 
                _,
                reservoir_usace_df,
                _,
            ) = nwm_forcing_preprocess(
                run_sets[run_set_iterator + 1],
                forcing_parameters,
                da_sets[run_set_iterator + 1] if data_assimilation_parameters else {},
                data_assimilation_parameters,
                break_network_at_waterbodies,
                segment_index,
                link_gage_df,
                usgs_lake_gage_crosswalk, 
                usace_lake_gage_crosswalk,
                link_lake_crosswalk,
                lastobs_df.index,
                cpu_pool,
                t0 + timedelta(seconds = dt * nts),
            )
            
            # if there are no TimeSlice files available for hybrid reservoir DA in the next loop, 
            # but there are DA parameters from the previous loop, then create a
            # dummy observations df. This allows the reservoir persistence to continue across loops.
            # USGS Reservoirs
            if not waterbody_types_df.empty:
                if 2 in waterbody_types_df['reservoir_type'].unique():
                    if reservoir_usgs_df.empty and len(reservoir_usgs_param_df.index) > 0:
                        reservoir_usgs_df = pd.DataFrame(
                            data    = np.nan, 
                            index   = reservoir_usgs_param_df.index, 
                            columns = [t0]
                        )

                # USACE Reservoirs   
                if 3 in waterbody_types_df['reservoir_type'].unique():
                    if reservoir_usace_df.empty and len(reservoir_usace_param_df.index) > 0:
                        reservoir_usace_df = pd.DataFrame(
                            data    = np.nan, 
                            index   = reservoir_usgs_param_df.index, 
                            columns = [t0]
                        )

                # update RFC lookback hours if there are RFC-type reservoirs in the simulation domain
                if 4 in waterbody_types_df['reservoir_type'].unique():
                    waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)     

            if showtiming:
                forcing_end_time = time.time()
                task_times['forcing_time'] += forcing_end_time - route_end_time
  
            if showtiming:
                ic_end_time = time.time()
                task_times['initial_condition_time'] += ic_end_time - forcing_end_time

        if showtiming:
            ic_start_time = time.time()
        
        # if streamflow DA is ON, then create a new lastobs dataframe
        if data_assimilation_parameters:
            streamflow_da = data_assimilation_parameters.get('streamflow_da',False)
            if streamflow_da:
                if streamflow_da.get('streamflow_nudging', False):
                    lastobs_df = new_lastobs(run_results, dt * nts)
            
        if showtiming:
            ic_end_time = time.time()
            task_times['initial_condition_time'] += ic_end_time - ic_start_time
        
        nwm_output_generator(
            run,
            run_results,
            supernetwork_parameters,
            output_parameters,
            parity_parameters,
            restart_parameters,
            parity_sets[run_set_iterator] if parity_parameters else {},
            qts_subdivisions,
            compute_parameters.get("return_courant", False),
            cpu_pool,
            waterbodies_df,
            waterbody_types_df,
            data_assimilation_parameters,
            lastobs_df,
            link_gage_df,
            link_lake_crosswalk,
        )
        
        if showtiming:
            output_end_time = time.time()
            task_times['output_time'] += output_end_time - ic_end_time
            
    if showtiming:
        task_times['total_time'] = time.time() - main_start_time

    LOG.debug("process complete in %s seconds." % (time.time() - main_start_time))
    
    if showtiming:
        print('************ TIMING SUMMARY ************')
        print('----------------------------------------')
        print(
            'Network graph construction: {} secs, {} %'\
            .format(
                round(task_times['network_time'],2),
                round(task_times['network_time']/task_times['total_time'] * 100,2)
            )
        )
        print(
            'Initial condition handling: {} secs, {} %'\
            .format(
                round(task_times['initial_condition_time'],2),
                round(task_times['initial_condition_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Forcing array construction: {} secs, {} %'\
            .format(
                round(task_times['forcing_time'],2),
                round(task_times['forcing_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Routing computations: {} secs, {} %'\
            .format(
                round(task_times['route_time'],2),
                round(task_times['route_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Output writing: {} secs, {} %'\
            .format(
                round(task_times['output_time'],2),
                round(task_times['output_time']/task_times['total_time'] * 100,2)
            )
        )
        print('----------------------------------------')
        print(
            'Total execution time: {} secs'\
            .format(
                round(task_times['network_time'],2) +
                round(task_times['initial_condition_time'],2) +
                round(task_times['forcing_time'],2) +
                round(task_times['route_time'],2) +
                round(task_times['output_time'],2)
            )
        )

async def main_v03_async(argv):
    """
    Handles the creation of the input parameter dictionaries
    from an input file and then sequences the execution of the
    t-route routing agorithm on a series of execution loops.
    """
    args = _handle_args_v03(argv)  # async shares input framework with non-async
    (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        diffusive_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)

    showtiming = log_parameters.get("showtiming", None)

    
    main_start_time = time.time()

    if preprocessing_parameters.get('use_preprocessed_data', False): 
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
        ) = unpack_nwm_preprocess_data(
            preprocessing_parameters
        )
    else:
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
        ) = nwm_network_preprocess(
            supernetwork_parameters,
            waterbody_parameters,
            preprocessing_parameters,
        )

    # TODO: This function modifies one of its arguments (waterbodies_df), which is somewhat poor practice given its otherwise functional nature. Consider refactoring
    waterbodies_df, q0, t0, lastobs_df, da_parameter_dict = nwm_initial_warmstate_preprocess(
        break_network_at_waterbodies,
        restart_parameters,
        data_assimilation_parameters,
        param_df.index,
        waterbodies_df,
        segment_list=None,
        wbodies_list=None,
    )

    # Create run_sets: sets of forcing files for each loop
    run_sets = nnu.build_forcing_sets(forcing_parameters, t0)

    # Create da_sets: sets of TimeSlice files for each loop
    if "data_assimilation_parameters" in compute_parameters: 
        da_sets = nnu.build_da_sets(data_assimilation_parameters, run_sets, t0)
        
    # Create parity_sets: sets of CHRTOUT files against which to compare t-route flows
    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
    else:
        parity_sets = []

    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    # TODO: Determine parameterization of the CPU and Threading pools
    # TODO: Make sure default values from dict.get for pool sizes work
    # e.g., is this valid: `ThreadPoolExecutor(max_workers=None)`?
    COMPUTE_cpu_pool = compute_parameters.get("cpu_pool", None)
    # IO_cpu_pool = compute_parameters.get("cpu_pool_IO", None)
    IO_cpu_pool = COMPUTE_cpu_pool
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)

    FORCE_KERNEL_THREAD = False
    if FORCE_KERNEL_THREAD:
        pool_IO = None
        pool_Processing = None
    else:
        pool_IO = concurrent.futures.ThreadPoolExecutor(max_workers=IO_cpu_pool)
        pool_Processing = concurrent.futures.ThreadPoolExecutor(max_workers=COMPUTE_cpu_pool)

    loop = asyncio.get_running_loop()

    forcings_task = loop.run_in_executor(
        pool_IO,
        nwm_forcing_preprocess,
        run_sets[0],
        forcing_parameters,
        da_sets[0] if data_assimilation_parameters else {},
        data_assimilation_parameters,
        break_network_at_waterbodies,
        param_df.index,
        lastobs_df.index,
        IO_cpu_pool,
        t0,
    )

    run_set_iterator = 0
    for run_set_iterator, run in enumerate(run_sets[:-1]):
                    
        dt = forcing_parameters.get("dt", None)
        nts = run.get("nts")
        
        qlats, usgs_df = await forcings_task
        
        # TODO: confirm utility of visual parity check in async execution
        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        model_task = loop.run_in_executor(
            pool_Processing,
            nwm_route,
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            COMPUTE_cpu_pool,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            usgs_df,
            lastobs_df,
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_parameters,
        )

        forcings_task = loop.run_in_executor(
            pool_IO,
            nwm_forcing_preprocess,
            run_sets[run_set_iterator + 1],
            forcing_parameters,
            da_sets[run_set_iterator + 1] if data_assimilation_parameters else {},
            data_assimilation_parameters,
            break_network_at_waterbodies,
            param_df.index,
            lastobs_df.index,
            IO_cpu_pool,
            t0 + timedelta(seconds = dt * nts),
        )

        run_results = await model_task

        q0 = new_nwm_q0(run_results)

        if data_assimilation_parameters:
            lastobs_df = new_lastobs(run_results, dt * nts)

        # TODO: Confirm this works with Waterbodies turned off
        waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

        if waterbody_type_specified:
            waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)

        output_task = loop.run_in_executor(
            pool_IO,
            nwm_output_generator,
            run,
            run_results,
            supernetwork_parameters,
            output_parameters,
            parity_parameters,
            restart_parameters,
            parity_sets[run_set_iterator] if parity_parameters else {},
            qts_subdivisions,
            compute_parameters.get("return_courant", False),
            IO_cpu_pool,
            data_assimilation_parameters,
            lastobs_df,
            link_gage_df,
        )

    # For the last loop, no next forcing or warm state is needed for execution.
    run_set_iterator += 1
    run = run_sets[run_set_iterator]

    dt = forcing_parameters.get("dt", None)
    nts = run.get("nts")

    qlats, usgs_df = await forcings_task

    # TODO: confirm utility of visual parity check in async execution
    if parity_sets:
        parity_sets[run_set_iterator]["dt"] = dt
        parity_sets[run_set_iterator]["nts"] = nts

    model_task = loop.run_in_executor(
        pool_Processing,
        nwm_route,
        connections,
        rconn,
        wbody_conn,
        reaches_bytw,
        parallel_compute_method,
        compute_kernel,
        subnetwork_target_size,
        COMPUTE_cpu_pool,
        dt,
        nts,
        qts_subdivisions,
        independent_networks,
        param_df,
        q0,
        qlats,
        usgs_df,
        lastobs_df,
        da_parameter_dict,
        assume_short_ts,
        return_courant,
        waterbodies_df,
        waterbody_parameters,
        waterbody_types_df,
        waterbody_type_specified,
        diffusive_parameters,
    )

    # nwm_final_output_generator()
    run_results = await model_task

    # These warmstates are never used for modeling, but
    # should be availble for last outputs.
    q0 = new_nwm_q0(run_results)

    if data_assimilation_parameters:
        lastobs_df = new_lastobs(run_results, dt * nts)

    waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

    if waterbody_type_specified:
        waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)

    output_task = await loop.run_in_executor(
        pool_IO,
        nwm_output_generator,
        run,
        run_results,
        supernetwork_parameters,
        output_parameters,
        parity_parameters,
        restart_parameters,
        parity_sets[run_set_iterator] if parity_parameters else {},
        qts_subdivisions,
        compute_parameters.get("return_courant", False),
        IO_cpu_pool,
        data_assimilation_parameters,
        lastobs_df,
        link_gage_df,
    )

    
    LOG.debug("process complete in %s seconds." % (time.time() - main_start_time))

    """
    Asynchronous execution Psuedocode
    Sync1: Prepare first warmstate from files
    Sync1: Prepare first forcing from files

    For first forcing set
        Sync2a: run model
        Sync2b: begin preparing next forcing
        Sync3a - AFTER Sync2a, prepare next warmstate (last state of model run)
        Sync3b: write any output from Sync2a
        Loop has to wait for Sync2a+b+Sync3a, does not have to wait for Sync3b
                if next forcing prepared
    """

    pool_IO.shutdown(wait=True)
    pool_Processing.shutdown(wait=True)


if __name__ == "__main__":
    v_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    v_parser.add_argument(
        "-V",
        "--input-version",
        default=3,
        nargs="?",
        choices=[2, 3, 4],
        type=int,
        help="Use version 2 or 3 of the input format. Default 3",
    )
    v_args = v_parser.parse_known_args()
    if v_args[0].input_version == 4:
        LOG.info("Running main v03 - async looping")
        coroutine = main_v03_async(v_args[1])
        asyncio.run(coroutine)
        # loop.run_until_complete(coroutine)
    if v_args[0].input_version == 3:
        LOG.info("Running main v03 - looping")
        main_v03(v_args[1])
