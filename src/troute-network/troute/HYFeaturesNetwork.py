from .AbstractNetwork import AbstractNetwork
import pandas as pd
import numpy as np
import geopandas as gpd
import time
import os
import json
from pathlib import Path

import troute.nhd_io as nhd_io #FIXME
import troute.hyfeature_preprocess as hyfeature_prep
from troute.nhd_network import reverse_dict, extract_connections

__verbose__ = False
__showtiming__ = False

def read_geopkg(file_path):
    flowpaths = gpd.read_file(file_path, layer="flowpaths")
    attributes = gpd.read_file(file_path, layer="flowpath_attributes").drop('geometry', axis=1)
    #merge all relevant data into a single dataframe
    flowpaths = pd.merge(flowpaths, attributes, on='id')

    return flowpaths

def read_json(file_path, edge_list):
    dfs = []
    with open(edge_list) as edge_file:
        edge_data = json.load(edge_file)
        edge_map = {}
        for id_dict in edge_data:
            edge_map[ id_dict['id'] ] = id_dict['toid']
        with open(file_path) as data_file:
            json_data = json.load(data_file)  
            for key_wb, value_params in json_data.items():
                df = pd.json_normalize(value_params)
                df['id'] = key_wb
                df['toid'] = edge_map[key_wb]
                dfs.append(df)
        df_main = pd.concat(dfs, ignore_index=True)

    return df_main

def numeric_id(flowpath):
    id = flowpath['id'].split('-')[-1]
    toid = flowpath['toid'].split('-')[-1]
    flowpath['id'] = int(id)
    flowpath['toid'] = int(toid)

def read_ngen_waterbody_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """
    Reads .gpkg or lake.json file and prepares a dataframe, filtered
    to the relevant reservoirs, to provide the parameters
    for level-pool reservoir computation.
    """
    def node_key_func(x):
        return int(x[3:])
    if os.path.splitext(parm_file)[1]=='.gpkg':
        df = gpd.read_file(parm_file, layer="lake_attributes").set_index('id')
    elif os.path.splitext(parm_file)[1]=='.json':
        df = pd.read_json(parm_file, orient="index")

    df.index = df.index.map(node_key_func)
    df.index.name = lake_index_field

    if lake_id_mask:
        df = df.loc[lake_id_mask]
    return df

def read_ngen_waterbody_type_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """
    """
    #FIXME: this function is likely not correct. Unclear how we will get 
    # reservoir type from the gpkg files. Information should be in 'crosswalk'
    # layer, but as of now (Nov 22, 2022) there doesn't seem to be a differentiation
    # between USGS reservoirs, USACE reservoirs, or RFC reservoirs...
    def node_key_func(x):
        return int(x[3:])
    
    if os.path.splitext(parm_file)[1]=='.gpkg':
        df = gpd.read_file(parm_file, layer="crosswalk").set_index('id')
    elif os.path.splitext(parm_file)[1]=='.json':
        df = pd.read_json(parm_file, orient="index")

    df.index = df.index.map(node_key_func)
    df.index.name = lake_index_field
    if lake_id_mask:
        df = df.loc[lake_id_mask]
        
    return df


class HYFeaturesNetwork(AbstractNetwork):
    """
    
    """
    __slots__ = []
    def __init__(self, 
                 supernetwork_parameters, 
                 waterbody_parameters,
                 data_assimilation_parameters,
                 restart_parameters=None, 
                 compute_parameters=None, 
                 verbose=False, 
                 showtiming=False):
        """
        
        """
        global __verbose__, __showtiming__
        __verbose__ = verbose
        __showtiming__ = showtiming
        if __verbose__:
            print("creating supernetwork connections set")
        if __showtiming__:
            start_time = time.time()
        
        #------------------------------------------------
        # Load Geo File
        #------------------------------------------------
        (self._dataframe,
         self._flowpath_dict,
         self._connections,
         self._waterbody_df,
         self._waterbody_types_df,
         self._terminal_codes,
        ) = hyfeature_prep.read_geo_file(
            supernetwork_parameters,
            waterbody_parameters,
        )
        
        #TODO Update for waterbodies and DA specific to HYFeatures...
        self._waterbody_connections = {}
        self._waterbody_type_specified = None
        self._gages = None
        self._link_lake_crosswalk = None


        if __verbose__:
            print("supernetwork connections set complete")
        if __showtiming__:
            print("... in %s seconds." % (time.time() - start_time))
            
        break_network_at_waterbodies = waterbody_parameters.get("break_network_at_waterbodies", False)        
        streamflow_da = data_assimilation_parameters.get('streamflow_da', False)
        break_network_at_gages       = False       
        if streamflow_da:
            break_network_at_gages   = streamflow_da.get('streamflow_nudging', False)
        break_points                 = {"break_network_at_waterbodies": break_network_at_waterbodies,
                                        "break_network_at_gages": break_network_at_gages}

        super().__init__(
            compute_parameters, 
            waterbody_parameters,
            restart_parameters,
            break_points,
            verbose=__verbose__,
            showtiming=__showtiming__,
            )   
            
        # Create empty dataframe for coastal_boundary_depth_df. This way we can check if
        # it exists, and only read in SCHISM data during 'assemble_forcings' if it doesn't
        self._coastal_boundary_depth_df = pd.DataFrame()

    def extract_waterbody_connections(rows, target_col, waterbody_null=-9999):
        """Extract waterbody mapping from dataframe.
        TODO deprecate in favor of waterbody_connections property"""
        return (
            rows.loc[rows[target_col] != waterbody_null, target_col].astype("int").to_dict()
        )

    @property
    def downstream_flowpath_dict(self):
        return self._flowpath_dict

    @property
    def waterbody_connections(self):
        """
            A dictionary where the keys are the reach/segment id, and the
            value is the id to look up waterbody parameters
        """
        if( not self._waterbody_connections ):
            #Funny story, NaN != NaN is true!!!!
            #Drop the nan, then check for waterbody_null just in case
            #waterbody_null happens to be NaN
            #FIXME this drops ALL nan, not just `waterbody`
            #waterbody_segments = self._dataframe.dropna().loc[
            #    self._dataframe["waterbody"] != self.waterbody_null, "waterbody"
            #]
            #waterbody_segments = waterbody_segments.loc[self.waterbody_dataframe.index]
            #self._waterbody_connections = waterbody_segments.index\
            #    .to_series(name = waterbody_segments.name)\
            #    .astype("int")\
            #    .to_dict()
            #If we identify as a waterbody, drop from the main dataframe
            #Ugh, but this drops everything that that MIGHT be a "lake"
            #without knowing if it was defined as a lake in the lake params
            #so this should just drop the waterbody_df index, not these segments...
            #In fact, these waterbody connections should probably be entirely reworked
            #with that in mind...
            self._waterbody_connections = self._waterbody_df.index.to_series(name = self._waterbody_df.index.name).astype("int").to_dict()
            #FIXME seems way more appropriate to do this in the constructor so the property doesn't side effect
            #the param df..., but then it breaks down the connection property...so for now, leave it here and fix later
            self._dataframe.drop(self._waterbody_df.index, axis=0, inplace=True)
        return self._waterbody_connections

    @property
    def gages(self):
        """
        FIXME
        """
        if self._gages is None and "gages" in self._dataframe.columns:
            self._gages = nhd_io.build_filtered_gage_df(self._dataframe[["gages"]])
        else:
            self._gages = {}
        return self._gages
    
    @property
    def waterbody_null(self):
        return np.nan #pd.NA
    
    def read_geo_file(
        self,
        supernetwork_parameters,
        waterbody_parameters,
    ):
        
        geo_file_path = supernetwork_parameters["geo_file_path"]
            
        file_type = Path(geo_file_path).suffix
        if(  file_type == '.gpkg' ):        
            self._dataframe = read_geopkg(geo_file_path)
        elif( file_type == '.json') :
            edge_list = supernetwork_parameters['flowpath_edge_list']
            self._dataframe = read_json(geo_file_path, edge_list) 
        else:
            raise RuntimeError("Unsupported file type: {}".format(file_type))

        # Don't need the string prefix anymore, drop it
        mask = ~ self.dataframe['toid'].str.startswith("tnex") 
        self._dataframe = self.dataframe.apply(numeric_id, axis=1)
            
        # make the flowpath linkage, ignore the terminal nexus
        self._flowpath_dict = dict(zip(self.dataframe.loc[mask].toid, self.dataframe.loc[mask].id))
        
        # **********  need to be included in flowpath_attributes  *************
        self._dataframe['alt'] = 1.0 #FIXME get the right value for this... 

        cols = supernetwork_parameters.get('columns',None)
        
        if cols:
            self._dataframe = self.dataframe[list(cols.values())]
            # Rename parameter columns to standard names: from route-link names
            #        key: "link"
            #        downstream: "to"
            #        dx: "Length"
            #        n: "n"  # TODO: rename to `manningn`
            #        ncc: "nCC"  # TODO: rename to `mannningncc`
            #        s0: "So"  # TODO: rename to `bedslope`
            #        bw: "BtmWdth"  # TODO: rename to `bottomwidth`
            #        waterbody: "NHDWaterbodyComID"
            #        gages: "gages"
            #        tw: "TopWdth"  # TODO: rename to `topwidth`
            #        twcc: "TopWdthCC"  # TODO: rename to `topwidthcc`
            #        alt: "alt"
            #        musk: "MusK"
            #        musx: "MusX"
            #        cs: "ChSlp"  # TODO: rename to `sideslope`
            self._dataframe = self.dataframe.rename(columns=reverse_dict(cols))
            self._dataframe.set_index("key", inplace=True)
            self._dataframe = self.dataframe.sort_index()

        # numeric code used to indicate network terminal segments
        terminal_code = supernetwork_parameters.get("terminal_code", 0)

        # There can be an externally determined terminal code -- that's this first value
        self._terminal_codes = set()
        self._terminal_codes.add(terminal_code)
        # ... but there may also be off-domain nodes that are not explicitly identified
        # but which are terminal (i.e., off-domain) as a result of a mask or some other
        # an interior domain truncation that results in a
        # otherwise valid node value being pointed to, but which is masked out or
        # being intentionally separated into another domain.
        self._terminal_codes = self.terminal_codes | set(
            self.dataframe[~self.dataframe["downstream"].isin(self.dataframe.index)]["downstream"].values
        )

        # build connections dictionary
        self._connections = extract_connections(
            self.dataframe, "downstream", terminal_codes=self.terminal_codes
        )

        #Load waterbody/reservoir info
        if waterbody_parameters:
            levelpool_params = waterbody_parameters.get('level_pool', None)
            if not levelpool_params:
                # FIXME should not be a hard requirement
                raise(RuntimeError("No supplied levelpool parameters in routing config"))
                
            lake_id = levelpool_params.get("level_pool_waterbody_id", "wb-id")
            self._waterbody_df = read_ngen_waterbody_df(
                        levelpool_params["level_pool_waterbody_parameter_file_path"],
                        lake_id,
                        )
                
            # Remove duplicate lake_ids and rows
            self._waterbody_df = (
                            self.waterbody_dataframe.reset_index()
                            .drop_duplicates(subset=lake_id)
                            .set_index(lake_id)
                            )

            try:
                self._waterbody_types_df = read_ngen_waterbody_type_df(
                                        levelpool_params["reservoir_parameter_file"],
                                        lake_id,
                                        #self.waterbody_connections.values(),
                                        )
                # Remove duplicate lake_ids and rows
                self._waterbody_types_df =(
                                    self.waterbody_types_dataframe.reset_index()
                                    .drop_duplicates(subset=lake_id)
                                    .set_index(lake_id)
                                    )

            except ValueError:
                #FIXME any reservoir operations requires some type
                #So make this default to 1 (levelpool)
                self._waterbody_types_df = pd.DataFrame(index=self.waterbody_dataframe.index)
                self._waterbody_types_df['reservoir_type'] = 1

