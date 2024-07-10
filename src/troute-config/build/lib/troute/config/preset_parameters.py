from pydantic import BaseModel

from typing_extensions import Literal

Presets = Literal["Standard_AnA","LongRange_AnA", "Extended_AnA",
                  "ShortRange_Forecast", "MediumRange_Forecast", "LongRange_Forecast", 
                  "Custom"]


class PresetParameters(BaseModel, extra='forbid'):
    """
    Predetermined configurations based on NWM operational modes. Presets are different
    durations of either retrospective (Analysis and Assimilation, AnA) or forecasts.

    This parameter is used in root_validators to set relevent compute and data assimliation
    parameters, simplifying required fields in configuration files.
    """
    # Predefined configuration names
    preset_configuration: Presets = "Custom"
