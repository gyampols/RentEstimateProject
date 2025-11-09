import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd  
import re
import numpy as np
from sklearn.neighbors import BallTree
import parameter_store

meta_cols = {"RegionID","SizeRank","RegionName","RegionType","StateName","State","City","Metro","CountyName"}

def preprocess_data(df):
    # close date should be converted to datetime
    df['Close Date'] = pd.to_datetime(df['Close Date'])
    # lets add a column for month of close date
    df['Close Month'] = df['Close Date'].dt.month
    df['Close Month End'] = df['Close Date'] + pd.offsets.MonthEnd(0)
    df=add_state_codes(df)
    df, month_cols=merge_zillow_data_by_zip(df)
    df=merge_zillow_data_by_state(df, month_cols)
    df.drop(columns=['Zipcode','State'], inplace=True)
    df=calc_distance_to_transit(df)
    return df


def add_state_codes(df):
    # Create a GeoDataFrame for the properties
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    # Load US states
    states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip')
    # Ensure both GeoDataFrames use the same CRS
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
    # Load Zillow ZIP-level data
    
    # Zillow ZIP-level dataset (one-bedroom) wide -> long + month-end merge
    zillow_data_zip = pd.read_csv('Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    # 1) Ensure df has a Zipcode column; derive from ZCTA if missing
    coord_mask = df[['Latitude', 'Longitude']].notna().all(axis=1)

    gdf_pts = gpd.GeoDataFrame(
        df.loc[coord_mask].copy(),
        geometry=[Point(xy) for xy in zip(df.loc[coord_mask, 'Longitude'], df.loc[coord_mask, 'Latitude'])],
        crs='EPSG:4326'
    )
    # Use the 500k generalized ZCTA shapefile (20m version 404s in this environment)
    zcta_url = 'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip'
    zcta = gpd.read_file(zcta_url)
    zcta = zcta.to_crs(gdf_pts.crs)
    cand_cols = ['ZCTA5CE20', 'ZCTA5CE10', 'GEOID', 'ZCTA5']
    zcta_col = next((c for c in cand_cols if c in zcta.columns), None)
    if zcta_col is None:
        raise ValueError(f"No ZCTA code column found in ZCTA layer. Columns: {list(zcta.columns)}")
    joined = gpd.sjoin(gdf_pts, zcta[[zcta_col, 'geometry']], how='left', predicate='intersects')
    df.loc[coord_mask, 'Zipcode'] = joined[zcta_col].astype(str).str.zfill(5).values

    # Normalize Zip format if present
    df['Zipcode'] = df.get('Zipcode', pd.Series([None]*len(df))).astype(str).str.zfill(5)

    # 2) Zillow file is wide -> melt to long
    month_cols = [c for c in zillow_data_zip.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]

    zillow_long = (
        zillow_data_zip
        .melt(id_vars=sorted(meta_cols & set(zillow_data_zip.columns)), value_vars=month_cols,
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
    # 4) Merge only if Zipcode appears to have non-null values
    if df['Zipcode'].notna().any():
        merged_df = pd.merge(
            df,
            zillow_long[['Zipcode','Date','ZillowValue']],
            left_on=['Zipcode','Close Month End'],
            right_on=['Zipcode','Date'],
            how='left'
        ).drop(columns=['Date'])
        match_rate = merged_df['ZillowValue'].notna().mean()
        print(f"Zillow match rate: {match_rate:.1%} ({merged_df['ZillowValue'].notna().sum()} / {len(merged_df)})")
    else:
        print("No valid Zipcode values found; skipping Zillow merge.")
        merged_df = df.copy()
        merged_df['ZillowValue'] = pd.NA

    return merged_df, month_cols


def merge_zillow_data_by_state(df, month_cols):
    
    # Zillow state-level dataset (one-bedroom) wide -> long + month-end merge
    zillow_data_state = pd.read_csv('State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    #we now pull in the zillow data by state and fill the missing zillow values according to state averages in a similar wayabs
    zillow_long_state = (
        zillow_data_state
        .melt(id_vars=sorted(meta_cols & set(zillow_data_state.columns)), value_vars=month_cols,
                var_name="Date", value_name="ZillowValue")
    )
    zillow_long_state['Date'] = pd.to_datetime(zillow_long_state['Date'])
    # Match RegionName which has the state name spelled out to its corresponding 2 letter state code. Ex: "California" -> "CA"
    state_name_to_code = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': '   AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
        'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
        'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
        'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',         
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    zillow_long_state['State'] = zillow_long_state['RegionName'].map(state_name_to_code)
    zillow_long_state = zillow_long_state.dropna(subset=['State'])
    # Merge on State and month-end Close Date
    merged_df_state = pd.merge(
        df,
        zillow_long_state[['State','Date','ZillowValue']],
        left_on=['State','Close Month End'],
        right_on=['State','Date'],
        how='left',
        suffixes=('', '_State')
    ).drop(columns=['Date'])
    # Fill missing ZIP-level Zillow values with state-level averages
    merged_df_state['ZillowValue_Filled'] = merged_df_state['ZillowValue']
    missing_zip_mask = merged_df_state['ZillowValue'].isna() & merged_df_state['ZillowValue_State'].notna()
    merged_df_state.loc[missing_zip_mask, 'ZillowValue_Filled'] = merged_df_state.loc[missing_zip_mask, 'ZillowValue_State']
    # then drop close month end, zillow value state and zillow value
    final_df = merged_df_state.drop(columns=['Close Month End', 'ZillowValue_State', 'ZillowValue'])
    # rename the filled column to ZHI   
    final_df.rename(columns={'ZillowValue_Filled': 'ZHI'}, inplace=True)
    return final_df


def calc_distance_to_transit(df):
    transit_cols = ['OBJECTID','stop_lat','stop_lon']
    transit_data = (
        pd.read_csv('NTAD_National_Transit_Map_Stops.csv', usecols=transit_cols)
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