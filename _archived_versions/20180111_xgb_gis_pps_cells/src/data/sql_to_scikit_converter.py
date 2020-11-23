import logging
import os
import pickle
import re
import uuid

import azure.storage as az
import fiona
import numpy as np
import pandas
from geopandas import GeoDataFrame
from osgeo.gdal import FileFromMemBuffer
from pandas import DataFrame
from sqlalchemy import create_engine

import util.config
from data import elb_repo, gis_repo

settings = util.config.load_settings()

logger = logging.getLogger(__name__)

pcsml_mssql_local_connstr = 'mssql+pyodbc://localhost/PcsML?driver=SQL+Server+Native+Client+11.0'
pcsml_psql_dkr_mbp_connstr = settings['db']['pcsml_pps_aggr']


def export_group_cols():
    """
    Run when the DB pps grouping columns change to create a new pickle.
    :return:
    """

    group_col_sql = """
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'PpsGrouping'
  AND (DATA_TYPE = 'varchar' OR COLUMN_NAME IN ('PlntDate#', 'HarvDate#'));
"""

    _engine = create_engine(pcsml_mssql_local_connstr)
    result = list(map(lambda r: r[0], _engine.execute(group_col_sql)))
    with open('data/group_cols.pickle', 'wb') as f:
        pickle.dump(result, f)


def group_cols():
    with open('data/group_cols.pickle', 'rb') as f:
        return pickle.load(f)


def load_corn_rows_mssql() -> pandas.DataFrame:
    data_query_sql = """
SELECT *
FROM PpsGrouping
WHERE
  crop IN ('corn', 'corn on corn')
  AND variety IS NOT NULL
  AND SoilGrtGroup IS NOT NULL
    AND [PlntDate#] IS NOT NULL
    AND [HarvDate#] IS NOT NULL
  AND Dry_Yield > 0
"""

    df = pandas.read_sql_query(data_query_sql, pcsml_mssql_local_connstr)
    return df


def load_corn_rows_psql() -> pandas.DataFrame:
    sql = """
SELECT *
FROM gis_pps_aggregate
WHERE ("Crop" = 'corn' OR "Crop" = 'corn on corn')
      AND "Variety" IS NOT NULL AND "Dry_Yield" IS NOT NULL
      AND "TotGdd" IS NOT NULL AND "TotRain" IS NOT NULL
      AND "SoilGrtGroup" IS NOT NULL
      AND "Lat" IS NOT NULL AND "Lon" IS NOT NULL
"""

    df = pandas.read_sql_query(sql, pcsml_psql_dkr_mbp_connstr)
    return df


def load_corn_rows_pickle_gz(version_token: str = '20170908'):
    df = pandas.read_pickle(f'data/pcs_ml_gis_groups_{version_token}.pickle.gz', compression='gzip')
    return df


def load_corn_rows_sample_shaped_pickle_gz(version_token: str = '20170910'):
    df = pandas.read_pickle(f'data/pcs_ml_gis_groups_sample_shaped_{version_token}.pickle.gz', compression='gzip')
    return df


def load_corn_data_frame():
    df = load_corn_rows_mssql()
    print(f"data shape: {df.shape}")

    # existence columns
    exist_cols_regex_prefixes = (
        'Disresis',
        'Foliar1', 'Foliar2', 'Foliar3',
        'Fung1', 'Fungicide1', 'Fung2', 'Fungicide2', 'Fung3', 'Fungicide3',
        'Isc1', 'Isc2', 'Isc3',
        'K2', r'K[A-Za-z]',
        'Lime',
        'Man',
        'Micro',
        'P2', r'P[A-Za-z]',
        'PostMethod',
        'Pre1', 'Pre2', 'Pre3', 'Pre6', 'Pre7', 'Pre8',
        'PreAppDate', 'PreMethod',
        'PreAdd1', 'PreAdd6',
        'PriN',
        'Pst1', 'Pst2',
        'Pst30', 'Pst31', 'Pst32', 'Pst33',
        'Pst40',
        r'Pst3[A-Za-z]',
        'PstAdd1', 'PstAdd2', 'PstAdd30', 'PstAdd40',
        'PstMethod2', 'PstMethod30', 'PstMethod40',
        'Pstresis',
        'Sc1N', 'Sc2N', 'Sc3N',
        'SeedTreat',
        'Spectrt',
        'StrType',
        r'Sulf[A-Za-z]', 'Sulf2'
    )
    exist_cols = [c for c in group_cols
                  if c not in ('PlntDate#', 'HarvDate#', 'PrevCrop', 'Crop')
                  and any(re.match(r, c) for r in exist_cols_regex_prefixes)]

    # for c in exist_cols:
    #     df['%s_exists' % c] = df[c].notnull()

    # dummy columns
    # ['PlntDate#', 'HarvDate#', 'SampleDate']
    dummy_cols_exclude = []
    logger.info("Final columns to exclude: %s", dummy_cols_exclude)
    dummy_cols = [col for col in group_cols if col not in dummy_cols_exclude]
    df = pandas.get_dummies(df, columns=dummy_cols)

    df.drop(['Year', 'YearId', 'ProcessedLayerUID'], axis=1, inplace=True)
    print(df.shape)
    # df.dropna(axis=1, how='all', inplace=True)
    # print(f"after na drop: {df.shape}")
    df.fillna(value=0, inplace=True)

    logger.info("DF final shape: %s" % str(df.shape))
    return df


def shape_pps_data(pps_df) -> pandas.DataFrame:
    # clean data values
    df: pandas.DataFrame = pps_df.apply(lambda x: x.astype(str).str.lower() if x.dtype == 'object' else x)
    df = df.replace('none', np.nan)
    df = df.replace('', np.nan)

    # existence columns
    existence_cols_regex_prefixes = (
        'Disresis',
        'Foliar1', 'Foliar2', 'Foliar3',
        'Fung1', 'Fungicide1', 'Fung2', 'Fungicide2', 'Fung3', 'Fungicide3',
        'Isc1', 'Isc2', 'Isc3',
        'K2', r'K[A-Za-z]',
        'Lime',
        'Man',
        'Micro',
        'P2', r'P[A-Za-z]',
        'PostMethod',
        'Pre1', 'Pre2', 'Pre3', 'Pre6', 'Pre7', 'Pre8',
        'PreAppDate', 'PreMethod',
        'PreAdd1', 'PreAdd6',
        'PriN',
        'Pst1', 'Pst2',
        'Pst30', 'Pst31', 'Pst32', 'Pst33',
        'Pst40',
        r'Pst3[A-Za-z]',
        'PstAdd1', 'PstAdd2', 'PstAdd30', 'PstAdd40',
        'PstMethod2', 'PstMethod30', 'PstMethod40',
        'Pstresis',
        'Sc1N', 'Sc2N', 'Sc3N',
        'SeedTreat',
        'Spectrt',
        'StrType',
        r'Sulf[A-Za-z]', 'Sulf2'
    )
    exist_cols = [c for c in group_cols()
                  if c not in ('PlntDate#', 'HarvDate#', 'PrevCrop', 'Crop')
                  and any(re.match(r, c) for r in existence_cols_regex_prefixes)]

    # for c in exist_cols:
    #     df['%s_exists' % c] = df[c].notnull()

    # dummy columns
    # ['PlntDate#', 'HarvDate#', 'SampleDate']
    dummy_cols_exclude = []
    logger.info("Final columns to exclude: %s", dummy_cols_exclude)
    dummy_cols = [col for col in group_cols() if col not in dummy_cols_exclude]
    df = pandas.get_dummies(df, columns=dummy_cols)

    df.drop(['Year', 'YearId', 'ProcessedLayerUID'], axis=1, inplace=True, errors='ignore')
    print(df.shape)
    # df.dropna(axis=1, how='all', inplace=True)
    # print(f"after na drop: {df.shape}")
    df.fillna(value=0, inplace=True)

    # remove spaces and invalid python variable chars from column names

    cols = df.columns.tolist()
    sanitized_cols = []
    for c in cols:
        c = re.sub('[^0-9a-zA-Z_]', '__', c)
        sanitized_cols.append(c)

    logging.info("sanatized columns:\n{}".format(sanitized_cols))
    df.columns = sanitized_cols

    logger.info("DF final shape: %s" % str(df.shape))
    return df


def dump_elbs(year=2016):
    storage_key = settings['azure']['pcs_storage_key']
    account = az.CloudStorageAccount(account_name='pcslive', account_key=storage_key)
    blob_service = account.create_block_blob_service()

    year_ids = elb_repo.get_elb_harvest_year_ids(year=2016)

    if not os.path.exists('data/elbs'): os.mkdir('data/elbs')

    for idx, elb_year_id in enumerate(year_ids):
        print("downloading elb GIS cells.  idx, yearid: ({} of {}), {}".format(idx, len(year_ids), elb_year_id))

        crop = gis_repo.get_pps_crop(elb_year_id)
        if not 'Corn' in crop:
            print("found not-corn crop, ignoring: {}".format(crop))
            continue

        # use the harvest layers
        elb_source_layers = [
            b.name
            for b in list(blob_service.list_blobs('sourcelayers', str(elb_year_id)))
            if any(x in b.name for x in ['_13_', '_14_', '_15_'])]
        elb_harvest_source_layer_name = elb_source_layers[0] if len(elb_source_layers) > 0 else None

        if elb_harvest_source_layer_name is None:
            print("ELB has no harvest layer: {}".format(elb_year_id))
            continue

        blob_zip = blob_service.get_blob_to_bytes('sourcelayers', elb_harvest_source_layer_name)

        vsiz = '/vsimem/{}.zip'.format(uuid.uuid4().hex)  # gdal/ogr requires a .zip extension
        FileFromMemBuffer(vsiz, bytes(blob_zip.content))
        with fiona.Collection(vsiz, vsi='zip') as f:
            shp = GeoDataFrame.from_features(f, crs={'init': 'epsg:4326'})

        elb_points = GeoDataFrame(shp.loc[shp['ELB_ID'] > 0])
        elb_centroids = list(elb_points.centroid)

        pps = gis_repo.processed_layer_shapes_by_year_id(elb_year_id)
        # get pps cells that have an elb
        pps_elb_cells = DataFrame(
            pps.loc[pps['geometry'].apply(lambda x: any(x.intersects(c) for c in elb_centroids))])
        pps_elb_cells.drop(['geometry'], inplace=True, axis=1)

        # load weather record
        wx = gis_repo.weather_by_year_id(elb_year_id)
        pps_elb_cells = pandas.concat([
            pps_elb_cells,
            pandas.DataFrame([wx.values], index=pps_elb_cells.index, columns=wx.keys())], axis=1)

        pps_elb_cells.to_pickle(f'data/elbs/{elb_year_id}_elb.pickle.gz', compression='gzip')


def load_cached_elbs(feature_names_ordered, data_path='data/elbs'):
    for f in [f for f in os.listdir(data_path) if "_elb.pickle.gz" in f]:
        tokens = f.split('_')
        feature_names_set = set(feature_names_ordered)

        elb = shape_pps_data(pandas.read_pickle(f'{data_path}/{f}', compression='gzip'))
        elb_cols = set(elb.columns.tolist())

        # add missing elb columns needed for the model
        missing_cols = feature_names_set - elb_cols
        elb: DataFrame = pandas.concat(
            [elb, pandas.DataFrame(0, index=elb.index, columns=missing_cols)], axis=1)

        # remove any extra enum dummy columns in elb (that training isn't aware of)
        elb_extra_cols = set(elb.columns) - feature_names_set
        if any(elb_extra_cols):
            print(f"WARNING: ELB has unknown training enum (dummy) cols: {','.join(elb_extra_cols)}")
            elb.drop(elb_extra_cols, axis=1, inplace=True)

        # re-order columns
        elb = elb[feature_names_ordered]

        elb_y = elb['Dry_Yield']
        elb_X = elb.drop(['Dry_Yield', 'Area'], axis=1)

        year_id = int(tokens[0])
        yield year_id, elb_X, elb_y, elb_extra_cols
