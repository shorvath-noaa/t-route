from abc import ABC, abstractmethod
import logging
import xarray as xr
import pandas as pd
import numpy as np
import time
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pathlib

import troute.nhd_io as nhd_io

LOG = logging.getLogger('')


def get_timesteps_from_nex(nexus_files):
    # Return a list of output files
    # Open the first nexus file and extract each timestamp
    output_file_timestamps = []
    with open(nexus_files[0]) as f:
        for line in f:
            output_file_timestamps.append(line.split(', ')[1])
    # Convert and reformat dates in the list
    output_file_timestamps = [pd.to_datetime(i).strftime("%Y%m%d%H%M") for i in output_file_timestamps]

    # Sort the list
    output_file_timestamps.sort()
    return output_file_timestamps


def split_csv_file(nexus_file, catchment_id, binary_folder):
    # Split the csv file into multiple csv files
    # Unescaped command: awk -F ', ' '{print "114085, "$NF >> "test/outputfile_"$1".txt"}' nex-114085_output.csv
    os.system(f'awk -F \', \' \'{{print "{catchment_id}, "$NF >> "{binary_folder}/tempfile_"$1".csv"}}\' {nexus_file}')


def rewrite_to_parquet(tempfile_id, output_file_id, binary_folder):
    # Rewrite the csv file to parquet
    df = pd.read_csv(f'{binary_folder}/tempfile_{tempfile_id}.csv', names=['feature_id', output_file_id])
    df.set_index('feature_id', inplace=True)  # Set feature_id as the index
    df[output_file_id] = df[output_file_id].astype(float)  # Convert output_file_id column to float64
    table_new = pa.Table.from_pandas(df)
    if not os.path.exists(f'{binary_folder}/{output_file_id}NEXOUT.parquet'):
        pq.write_table(table_new, f'{binary_folder}/{output_file_id}NEXOUT.parquet')
    else:
        raise Exception(f'Parquet file {binary_folder}/{output_file_id}NEXOUT.parquet already exists')

def nex_files_to_binary(nexus_files, binary_folder):
    # Get the output files
    output_timesteps = get_timesteps_from_nex(nexus_files)
    
    # Split the csv file into multiple csv files
    for nexus_file in nexus_files:
        catchment_id = get_id_from_filename(nexus_file)
        split_csv_file(nexus_file, catchment_id, binary_folder)
    
    # Rewrite the temp csv files to parquet
    for tempfile_id, nexus_file in enumerate(output_timesteps):
        rewrite_to_parquet(tempfile_id, nexus_file, binary_folder)
    
    # Clean up the temp files
    os.system(f'rm -rf {binary_folder}/tempfile_*.csv')
    
    nexus_input_folder = binary_folder
    forcing_glob_filter = '*NEXOUT.parquet'
    
    return nexus_input_folder, forcing_glob_filter

def get_id_from_filename(file_name):
    id = os.path.splitext(file_name)[0].split('-')[1].split('_')[0]
    return int(id)

def read_file(file_name):
    extension = file_name.suffix
    if extension=='.csv':
        df = pd.read_csv(file_name)
    elif extension=='.parquet':
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    elif extension=='.nc':
        df = xr.open_dataset(file_name).to_pandas().reset_index()
        df.index.name = None
    return df

def read_DFlow_output(ds):
    df = ds[['waterlevel','bedlevel']].to_dataframe()
    df['depth'] = df['waterlevel'] + df['bedlevel']
    df['station_name'] = df['station_name'].str.decode('utf-8').str.split(' / ',expand=True).loc[:,1].astype(float).astype(int)
    df = df.reset_index()[['time','station_name','depth']].set_index(['station_name', 'time']).unstack('time', fill_value = np.nan)['depth']
    return df

def read_SCHISM_output(ds):
    df = ds.to_dataframe()
    df['depth'] = df['depth'] + df['elev']
    df = df[['depth']].unstack('time', fill_value = np.nan)['depth']
    return df

def read_coastal_output(filepath):
    ds = xr.open_dataset(filepath)
    coastal_model_indicator = ds.attrs.get('institution')
    if coastal_model_indicator=='SCHISM Model output':
        df = read_SCHISM_output(ds)
    elif coastal_model_indicator=='Deltares':
        df = read_DFlow_output(ds)
    return df


class AbstractForcing(ABC):
    """
    
    """
    __slots__ = ["_forcing_parameters", "_supernetwork_parameters", "_run_sets"]
    
    def __init__(self, forcing_parameters, supernetwork_parameters,):
        """
        
        """
        self._forcing_parameters = forcing_parameters
        self._supernetwork_parameters = supernetwork_parameters

    def build_forcing_sets(self,):

        forcing_parameters = self._forcing_parameters
        supernetwork_parameters = self._supernetwork_parameters

        run_sets           = forcing_parameters.get("qlat_forcing_sets", None)
        qlat_input_folder  = forcing_parameters.get("qlat_input_folder", None)
        nts                = forcing_parameters.get("nts", None)
        max_loop_size      = forcing_parameters.get("max_loop_size", 12)
        dt                 = forcing_parameters.get("dt", None)

        try:
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            assert qlat_input_folder.is_dir() == True
        except TypeError:
            raise TypeError("Aborting simulation because no qlat_input_folder is specified in the forcing_parameters section of the .yaml control file.") from None
        except AssertionError:
            raise AssertionError("Aborting simulation because the qlat_input_folder:", qlat_input_folder,"does not exist. Please check the the nexus_input_folder variable is correctly entered in the .yaml control file") from None

        forcing_glob_filter = forcing_parameters["qlat_file_pattern_filter"]

        if forcing_glob_filter=="nex-*":
            print("Reformating qlat nexus files as hourly binary files...")
            binary_folder = forcing_parameters.get('binary_nexus_file_folder', None)
            qlat_files = qlat_input_folder.glob(forcing_glob_filter)

            #Check that directory/files specified will work
            if not binary_folder:
                raise(RuntimeError("No output binary qlat folder supplied in config"))
            elif not os.path.exists(binary_folder):
                raise(RuntimeError("Output binary qlat folder supplied in config does not exist"))
            elif len(list(pathlib.Path(binary_folder).glob('*.parquet'))) != 0:
                raise(RuntimeError("Output binary qlat folder supplied in config is not empty (already contains '.parquet' files)"))

            #Add tnx for backwards compatability
            qlat_files_list = list(qlat_files) + list(qlat_input_folder.glob('tnx*.csv'))
            #Convert files to binary hourly files, reset nexus input information
            qlat_input_folder, forcing_glob_filter = nex_files_to_binary(qlat_files_list, binary_folder)
            forcing_parameters["qlat_input_folder"] = qlat_input_folder
            forcing_parameters["qlat_file_pattern_filter"] = forcing_glob_filter
            
        # TODO: Throw errors if insufficient input data are available
        if run_sets:        
            #FIXME: Change it for hyfeature
            '''
            # append final_timestamp variable to each set_list
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            for (s, _) in enumerate(run_sets):
                final_chrtout = qlat_input_folder.joinpath(run_sets[s]['qlat_files'
                        ][-1])
                final_timestamp_str = nhd_io.get_param_str(final_chrtout,
                        'model_output_valid_time')
                run_sets[s]['final_timestamp'] = \
                    datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')
            '''  
        elif qlat_input_folder:        
            # Construct run_set dictionary from user-specified parameters

            # get the first and seconded files from an ordered list of all forcing files
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            all_files          = sorted(qlat_input_folder.glob(forcing_glob_filter))
            first_file         = all_files[0]
            second_file        = all_files[1]

            # Deduce the timeinterval of the forcing data from the output timestamps of the first
            # two ordered CHRTOUT files
            if forcing_glob_filter=="*.CHRTOUT_DOMAIN1":
                t1 = nhd_io.get_param_str(first_file, "model_output_valid_time")
                t1 = datetime.strptime(t1, "%Y-%m-%d_%H:%M:%S")
                t2 = nhd_io.get_param_str(second_file, "model_output_valid_time")
                t2 = datetime.strptime(t2, "%Y-%m-%d_%H:%M:%S")
            elif forcing_glob_filter.startswith('*NEXOUT'):
                t1_str = first_file.name.split('NEXOUT', 1)[0]
                t1 = datetime.strptime(t1_str, '%Y%m%d%H%M')
                t2_str = second_file.name.split('NEXOUT', 1)[0]
                t2 = datetime.strptime(t2_str, '%Y%m%d%H%M')
            else:
                df     = read_file(first_file)
                t1_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
                t1     = datetime.strptime(t1_str,"%Y-%m-%d_%H:%M:%S")
                df     = read_file(second_file)
                t2_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
                t2     = datetime.strptime(t2_str,"%Y-%m-%d_%H:%M:%S")
            
            dt_qlat_timedelta = t2 - t1
            dt_qlat = dt_qlat_timedelta.seconds

            # determine qts_subdivisions
            qts_subdivisions = dt_qlat / dt
            if dt_qlat % dt == 0:
                qts_subdivisions = int(dt_qlat / dt)
            # make sure that qts_subdivisions = dt_qlat / dt
            forcing_parameters['qts_subdivisions']= qts_subdivisions

            # the number of files required for the simulation
            nfiles = int(np.ceil(nts / qts_subdivisions))
            
            # list of forcing file datetimes
            #datetime_list = [t0 + dt_qlat_timedelta * (n + 1) for n in
            #                 range(nfiles)]
            # ** Correction ** Because qlat file at time t is constantly applied throughout [t, t+1],
            #               ** n + 1 should be replaced by n
            datetime_list = [self.t0 + dt_qlat_timedelta * (n) for n in
                            range(nfiles)]        
            datetime_list_str = [datetime.strftime(d, '%Y%m%d%H%M') for d in
                                datetime_list]

            # list of forcing files
            forcing_filename_list = [d_str + forcing_glob_filter[1:] for d_str in
                                    datetime_list_str]
            
            # check that all forcing files exist
            for f in forcing_filename_list:
                try:
                    J = pathlib.Path(qlat_input_folder.joinpath(f))     
                    assert J.is_file() == True
                except AssertionError:
                    raise AssertionError("Aborting simulation because forcing file", J, "cannot be not found.") from None
                    
            # build run sets list
            run_sets = []
            k = 0
            j = 0
            nts_accum = 0
            nts_last = 0
            while k < len(forcing_filename_list):
                run_sets.append({})

                if k + max_loop_size < len(forcing_filename_list):
                    run_sets[j]['qlat_files'] = forcing_filename_list[k:k
                        + max_loop_size]
                else:
                    run_sets[j]['qlat_files'] = forcing_filename_list[k:]

                nts_accum += len(run_sets[j]['qlat_files']) * qts_subdivisions
                if nts_accum <= nts:
                    run_sets[j]['nts'] = int(len(run_sets[j]['qlat_files'])
                                            * qts_subdivisions)
                else:
                    run_sets[j]['nts'] = int(nts - nts_last)

                final_qlat = qlat_input_folder.joinpath(run_sets[j]['qlat_files'][-1]) 
                if forcing_glob_filter=="*.CHRTOUT_DOMAIN1":           
                    final_timestamp_str = nhd_io.get_param_str(final_qlat,'model_output_valid_time')
                elif forcing_glob_filter.startswith('*NEXOUT'):
                    
                    final_timestamp_str = datetime.strptime(
                        final_qlat.name.split('NEXOUT', 1)[0],
                        "%Y%m%d%H%M"
                    ).strftime("%Y-%m-%d_%H:%M:%S")
                else:
                    df = read_file(final_qlat)
                    final_timestamp_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")           
                
                run_sets[j]['final_timestamp'] = \
                    datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')

                nts_last = nts_accum
                k += max_loop_size
                j += 1

        return run_sets
    
    
    
    def assemble_forcings(self, run,):
        """
        Assemble model forcings. Forcings include hydrological lateral inflows (qlats)
        and coastal boundary depths for hybrid runs
        
        Aguments
        --------
        - run                          (dict): List of forcing files pertaining to a 
                                               single run-set

        Returns
        -------
        
        Notes
        -----
        
        """
    
        # Unpack user-specified forcing parameters
        dt                           = self.forcing_parameters.get("dt", None)
        qts_subdivisions             = self.forcing_parameters.get("qts_subdivisions", None)
        qlat_input_folder            = self.forcing_parameters.get("qlat_input_folder", None)
        qlat_file_index_col          = self.forcing_parameters.get("qlat_file_index_col", "feature_id")
        qlat_file_value_col          = self.forcing_parameters.get("qlat_file_value_col", "q_lateral")
        qlat_file_gw_bucket_flux_col = self.forcing_parameters.get("qlat_file_gw_bucket_flux_col", "qBucket")
        qlat_file_terrain_runoff_col = self.forcing_parameters.get("qlat_file_terrain_runoff_col", "qSfcLatRunoff")

    
        # TODO: find a better way to deal with these defaults and overrides.
        run["t0"]                           = run.get("t0", self.t0)
        run["dt"]                           = run.get("dt", dt)
        run["qts_subdivisions"]             = run.get("qts_subdivisions", qts_subdivisions)
        run["qlat_input_folder"]            = run.get("qlat_input_folder", qlat_input_folder)
        run["qlat_file_index_col"]          = run.get("qlat_file_index_col", qlat_file_index_col)
        run["qlat_file_value_col"]          = run.get("qlat_file_value_col", qlat_file_value_col)
        run["qlat_file_gw_bucket_flux_col"] = run.get("qlat_file_gw_bucket_flux_col", qlat_file_gw_bucket_flux_col)
        run["qlat_file_terrain_runoff_col"] = run.get("qlat_file_terrain_runoff_col", qlat_file_terrain_runoff_col)
        
        #---------------------------------------------------------------------------
        # Assemble lateral inflow data
        #---------------------------------------------------------------------------

        # Place holder, if reading qlats from a file use this.
        # TODO: add an option for reading qlat data from BMI/model engine
        start_time = time.time()
        LOG.info("Creating a DataFrame of lateral inflow forcings ...")

        self.build_qlateral_array(
            run,
        )
        
        LOG.debug(
            "lateral inflow DataFrame creation complete in %s seconds." \
                % (time.time() - start_time)
                )

        #---------------------------------------------------------------------
        # Assemble coastal coupling data [WIP]
        #---------------------------------------------------------------------
        # Run if coastal_boundary_depth_df has not already been created:
        if self._coastal_boundary_depth_df.empty:
            coastal_boundary_elev_files = self.forcing_parameters.get('coastal_boundary_input_file', None) 
            coastal_boundary_domain_files = self.hybrid_parameters.get('coastal_boundary_domain', None)    
            
            if coastal_boundary_elev_files:
                #start_time = time.time()    
                #LOG.info("creating coastal dataframe ...")
                
                coastal_boundary_domain   = nhd_io.read_coastal_boundary_domain(coastal_boundary_domain_files)          
                self._coastal_boundary_depth_df = read_coastal_output(coastal_boundary_elev_files)
                    
                #LOG.debug(
                #    "coastal boundary elevation observation DataFrame creation complete in %s seconds." \
                #    % (time.time() - start_time)
                #)
            else:
                self._coastal_boundary_depth_df = pd.DataFrame()
    
