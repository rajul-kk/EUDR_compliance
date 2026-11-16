"""
Configuration file for GEE assets.
"""

HANSEN_ASSET_ID = 'UMD/hansen/global_forest_change_2025_v1_13'
DYNAMIC_WORLD_ASSET_ID = 'GOOGLE/DYNAMICWORLD/V1'      # kept for reference
CANOPY_HEIGHT_ASSET_ID = 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1'  # kept for reference

HEIGHT_THRESHOLD = 5  # meters
FOREST_LABEL = 1
CROP_LABEL = 2

DW_TREES_LABEL = 1
DW_CROPS_LABEL = 4
DW_SHRUB_LABEL = 5

CLASS_FOREST = 1
CLASS_CROPS = 2
CLASS_SHRUB = 3
CLASS_OTHER = 0
