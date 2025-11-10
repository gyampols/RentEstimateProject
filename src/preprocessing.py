import os
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import re
import numpy as np
from sklearn.neighbors import BallTree
from src import parameter_store as ps

# Resolve project root relative to this file so data paths work in CI and locally
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def preprocess_data(df):
    """
    End-to-end preprocessing without using Close Date. Computes ZHI as the
    average Zillow index over months in 2023 and 2024, preferring ZIP-level
    averages and falling back to state-level averages when ZIP is unavailable.
    """
    df = add_state_codes(df)
    df = merge_zillow_data_by_zip(df)
    df = merge_zillow_data_by_state(df)
    # Cleanup helper columns
    df=df.drop(columns=['Zipcode', 'State'], errors='ignore')
    df = calc_distance_to_transit(df)
    return df


def add_state_codes(df):
    # Create a GeoDataFrame for the properties
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    # Load US states
    states = gpd.read_file(ps.state_codes_crs_url)
    # Ensure both GeoDataFrames use the same CRS
    # Guard against missing CRS on states (rare but defensively handle)
    if states.crs is not None:
        gdf = gdf.to_crs(states.crs)
    # Perform spatial join to get state information for each property
    gdf_with_state = gpd.sjoin(gdf, states[['STUSPS', 'geometry']], how='left')
    # Rename the state code column
    gdf_with_state.rename(columns={'STUSPS': 'State'}, inplace=True)
    # Now gdf_with_state has a 'State' column with the state code for each property.
    # If you want to convert it back to a regular DataFrame without geometry:
    df_with_state = pd.DataFrame(gdf_with_state.drop(columns='geometry'))
    df['State'] = df_with_state['State']
    
    return df


def merge_zillow_data_by_zip(df):
    """Attach ZIP-level ZHI computed as the mean of 2023–2024 Zillow indices.
    Ensures df['Zipcode'] exists (derived from coordinates via ZCTA overlay).
    """
    # Ensure df has a Zipcode column; derive from ZCTA if missing
    coord_mask = df[['Latitude', 'Longitude']].notna().all(axis=1)

    gdf_pts = gpd.GeoDataFrame(
        df.loc[coord_mask].copy(),
        geometry=[Point(xy) for xy in zip(df.loc[coord_mask, 'Longitude'], df.loc[coord_mask, 'Latitude'])],
        crs='EPSG:4326'
    )
    # Use the 500k generalized ZCTA shapefile
    zcta = gpd.read_file(ps.zip_codes_crs_url)
    if gdf_pts.crs is not None and zcta.crs is not None:
        zcta = zcta.to_crs(gdf_pts.crs)
    cand_cols = ['ZCTA5CE20', 'ZCTA5CE10', 'GEOID', 'ZCTA5']
    zcta_col = next((c for c in cand_cols if c in zcta.columns), None)
    if zcta_col is None:
        raise ValueError(f"No ZCTA code column found in ZCTA layer. Columns: {list(zcta.columns)}")
    joined = gpd.sjoin(gdf_pts, zcta[[zcta_col, 'geometry']], how='left', predicate='intersects')
    df.loc[coord_mask, 'Zipcode'] = joined[zcta_col].astype(str).str.zfill(5).values

    # Normalize Zip format if present
    df['Zipcode'] = df.get('Zipcode', pd.Series([None]*len(df))).astype(str).str.zfill(5)

    # Load Zillow ZIP-level data and compute 2023–2024 average per ZIP
    zillow_zip_path = os.path.join(DATA_DIR, 'Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    zillow_data_zip = pd.read_csv(zillow_zip_path)
    month_cols = [c for c in zillow_data_zip.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    zillow_long = (
        zillow_data_zip
        .melt(id_vars=sorted(ps.meta_cols & set(zillow_data_zip.columns)), value_vars=month_cols,
              var_name="Date", value_name="ZillowValue")
    )
    zillow_long['Date'] = pd.to_datetime(zillow_long['Date'])
    # Derive Zipcode from RegionName where RegionType == 'zip' and zero-pad
    if 'RegionType' in zillow_long.columns and 'RegionName' in zillow_long.columns:
        zip_mask = zillow_long['RegionType'].str.lower() == 'zip'
        zillow_long.loc[zip_mask, 'Zipcode'] = zillow_long.loc[zip_mask, 'RegionName'].astype(str).str.zfill(5)
    else:
        raise ValueError("Missing RegionType/RegionName for ZIP derivation.")
    zillow_long = zillow_long.dropna(subset=['Zipcode'])

    # Filter to 2023 and 2024 inclusive
    yr_mask = (zillow_long['Date'].dt.year >= 2023) & (zillow_long['Date'].dt.year <= 2024)
    zhi_zip = zillow_long.loc[yr_mask].groupby('Zipcode', as_index=False)['ZillowValue'].mean()
    # Rename without using rename() to satisfy strict type checks
    zhi_zip.columns = ['Zipcode', 'ZHI_Zip']

    # Merge ZHI_Zip to df
    df = df.merge(zhi_zip, on='Zipcode', how='left')
    return df


def merge_zillow_data_by_state(df):
    """Attach State-level ZHI for 2023–2024 and finalize ZHI with ZIP fallback.
    Creates final 'ZHI' column: prefer ZHI_Zip where available else ZHI_State.
    """
    # Load Zillow state-level data and compute 2023–2024 average per state
    zillow_state_path = os.path.join(DATA_DIR, 'State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    zillow_data_state = pd.read_csv(zillow_state_path)
    month_cols = [c for c in zillow_data_state.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    zillow_long_state = (
        zillow_data_state
        .melt(id_vars=sorted(ps.meta_cols & set(zillow_data_state.columns)), value_vars=month_cols,
              var_name="Date", value_name="ZillowValue")
    )
    zillow_long_state['Date'] = pd.to_datetime(zillow_long_state['Date'])
    # Map state full name to two-letter code
    zillow_long_state['State'] = zillow_long_state['RegionName'].map(ps.state_name_to_code)
    zillow_long_state = zillow_long_state.dropna(subset=['State'])

    yr_mask = (zillow_long_state['Date'].dt.year >= 2023) & (zillow_long_state['Date'].dt.year <= 2024)
    zhi_state = zillow_long_state.loc[yr_mask].groupby('State', as_index=False)['ZillowValue'].mean()
    zhi_state.columns = ['State', 'ZHI_State']

    df = df.merge(zhi_state, on='State', how='left')
    # Final ZHI selection: prefer ZIP, fallback to State
    df['ZHI'] = df['ZHI_Zip']
    use_state = df['ZHI'].isna() & df['ZHI_State'].notna()
    df.loc[use_state, 'ZHI'] = df.loc[use_state, 'ZHI_State']
    # Drop helper columns
    df.drop(columns=['ZHI_Zip', 'ZHI_State'], inplace=True, errors='ignore')
    return df


def calc_distance_to_transit(df):
    transit_cols = ['OBJECTID','stop_lat','stop_lon']
    transit_path = os.path.join(DATA_DIR, 'NTAD_National_Transit_Map_Stops.csv')
    transit_data = (
        pd.read_csv(transit_path, usecols=transit_cols)
        .dropna(subset=['stop_lat','stop_lon'])
    )

    # Filter valid property coordinates
    coord_mask = df[['Latitude','Longitude']].notna().all(axis=1)
    valid_df = df.loc[coord_mask, ['Latitude','Longitude']].copy()


    # Convert degrees -> radians for haversine BallTree
    prop_rad = np.radians(valid_df[['Latitude','Longitude']].values)
    stops_rad = np.radians(transit_data[['stop_lat','stop_lon']].values)

    # Build BallTree (haversine distances on unit sphere)
    tree = BallTree(stops_rad, metric='haversine')

    # Query nearest stop (returns distance in radians); multiply by earth radius (meters)
    earth_radius_m = 6371000.0
    dist_rad, _ = tree.query(prop_rad, k=1)
    dist_m = dist_rad.flatten() * earth_radius_m
    # Assign back
    df.loc[coord_mask, 'DistanceToTransit'] = dist_m
    df.loc[~coord_mask, 'DistanceToTransit'] = np.nan
    return df