import numpy as np
import logging
LOG = logging.getLogger('')

def reservoir_hybrid_da(
    lake_number,
    gage_obs,
    gage_time,
    now,
    previous_persisted_outflow,
    persistence_update_time,
    persistence_index,
    climatology_outflow,
    obs_lookback_hours,
    update_time,
    update_time_interval = 3600,
    persistence_update_time_interval = 86400,
):

    """
    Perform persistence reservoir data assimilation for the Great Lakes.
    
    Arguments
    ---------
    - lake_number                          (int): unique lake identification
                                                  number
    - gage_obs                (memoryview slice): array of gage observations for
                                                  at the gage associated with this
                                                  waterbody
    - gage_time               (memoryview slice): array of observation times
                                                  (secs) relative to the model
                                                  initialization time (t0).
    - now                                (float): Current time, seconds since in-
                                                  itialization time (t0).
    - previous_persisted_outflow (numpy.float32): Persisted outflow value from 
                                                  last timestep.                                
    - obs_lookback_hours                   (int): Maximum allowable lookback time
                                                  when searching for new outflow
                                                  observations
    - update_time                (numpy.float32): time to search for new observ-
                                                  ations, seconds since model 
                                                  initialization time (t0)
    - update_time_interval               (float): Time interval from current to
                                                  next update time (secs)
    
    Returns
    -------
    - outflow             (float): Persisted reservoir outflow rate (m3/sec)
    """

    LOG.debug('Great Lakes data assimilation for lake_id: %s at time %s from run start' % (lake_number, now))
    
    # set persistence limit at persistence_update_time cycles
    persistence_limit = 11
    
    # initialize new_persistence_index as persistence_index and 
    # new_persistence_update_time as persistence_update_time. 
    new_persistence_index = persistence_index
    new_persistence_update_time = persistence_update_time
    
    # initialize new_update_time as update time. If the update time needs to be
    # updated, then this variable will be reset later.
    new_update_time = update_time
    
    if now >= update_time:
        LOG.debug(
            'Looking for observation to assimilate...'
        )
    
        # initialize variable to store assimilated observations. We initialize
        # as np.nan, so that if no good quality observations are found, we can
        # easily catch it.
        obs = np.nan
        
        # identify TimeSlice time (gage_time) index nearest to, but not greater 
        # than the update_time
        t_diff = update_time - gage_time
        t_idx = np.where(t_diff >=  0, t_diff, np.inf).argmin()
        
        # look backwards from the nearest update_time for the first available 
        # observation,
        # NOTE: QA/QC has already happened upstream upon loading and formatting 
        # TimeSlice observations, so all poor values have already been replaced 
        # with nans
        for i in range(t_idx, -1, -1):
            
            # check if gage observation is good quality (not nan)
            if np.isnan(gage_obs[i]) == False:
                
                # record good observation to obs
                obs = gage_obs[i]
                
                # determine how many seconds prior to the update_time the 
                # observation was taken
                t_obs = gage_time[i]
                gage_lookback_seconds = update_time - t_obs
                
                # reset the observation update time
                new_update_time = update_time + update_time_interval
                
                break
                
        if np.isnan(obs): # no good observation was found
        
            '''
            - If no good observation is found, then we do not update the 
              update time. Consequently we will continue to search for a good
              observation at each model timestep, before updating the update
              time. 

            '''
            LOG.debug(
                'No good observation found, persisting previously assimilated flow'
            )
            persisted_outflow = previous_persisted_outflow
            
            if now >= persistence_update_time:
                new_persistence_index = persistence_index + 1
                new_persistence_update_time = persistence_update_time \
                    + persistence_update_time_interval

        
        else: # good observation found    
        
            # check that observation is not taken from beyond the
            # allowable lookback window
            if gage_lookback_seconds > obs_lookback_hours*60*60:
                LOG.debug('good observation found, but is outside of lookback window')
                LOG.debug(
                    'observation at %s seconds from update time', 
                    gage_lookback_seconds
                )
                persisted_outflow = previous_persisted_outflow
                if now >= persistence_update_time:
                    new_persistence_index = persistence_index + 1
                    new_persistence_update_time = persistence_update_time \
                        + persistence_update_time_interval
                
            else:
                LOG.debug('good observation found!: %s cms', obs)
                LOG.debug(
                    'observation at %s seconds from update time', 
                    gage_lookback_seconds
                )
                # the new persisted outflow is the discovered gage observation
                persisted_outflow = obs
                # reset persistence index and update persistence update time
                new_persistence_index = 1
                new_persistence_update_time = persistence_update_time \
                    + persistence_update_time_interval
    
    elif now >= persistence_update_time:
     
        # increment the persistence index
        new_persistence_index = persistence_index + 1
        new_persistence_update_time = persistence_update_time \
            + persistence_update_time_interval
            
        # persist previously persisted outflow value
        if persistence_index <= persistence_limit:
            LOG.debug(
                'Persisting previously assimilated outflow'
            )
            persisted_outflow = previous_persisted_outflow
            
        # persistence limit reached - use levelpool outflow
        if persistence_index > persistence_limit:
            LOG.debug(
                'Persistence limit reached, defaulting to climatological outflow'
            )
            persisted_outflow = climatology_outflow
            new_persistence_index = 0
        
    else:
        LOG.debug(
            'Persisting previously assimilated outflow'
        )
        persisted_outflow = previous_persisted_outflow
    
    # set reservoir outflow
    if np.isnan(persisted_outflow):
        LOG.debug(
            'Previously persisted outflow is nan, defaulting to climatological outflow'
        )
        # levelpool outflow
        outflow = climatology_outflow
        new_persistence_index = 0
    
    else:
        # data assimilated outflow
        outflow = persisted_outflow

    return outflow, persisted_outflow, new_update_time, new_persistence_index, new_persistence_update_time
