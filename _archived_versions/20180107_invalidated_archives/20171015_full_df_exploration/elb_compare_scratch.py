import pickle
import uuid
from datetime import datetime

import azure.storage as az
import fiona
import pandas
import sqlalchemy as sql
from geopandas import GeoDataFrame
from osgeo.gdal import FileFromMemBuffer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

import data_scripts.elb_repo as elb_repo
import data_scripts.gis_repo as gis_repo
from data_scripts import pcs_data_loader
from modeling import score_util
from util import config

# test yearid: 303607

settings = config.load_settings()

db = sql.create_engine(settings['db']['gis'])

storage_sas = settings['azure']['pcs_storage_sas']
storage_key = settings['azure']['pcs_storage_key']

# load training data and train et model
train_df = pcs_data_loader.load_corn_data_frame()
train_cols = set(train_df.keys())
y = train_df['Dry_Yield']
X = train_df.drop(['Dry_Yield', 'Area'], axis=1)
scaler = StandardScaler()
scaler.fit(X)
extra_trees = ExtraTreesRegressor(n_jobs=-1, verbose=True, n_estimators=45)
extra_trees.fit(scaler.transform(X), y)

# account = az.CloudStorageAccount(account_name='pcslive', sas_token=storage_sas)
account = az.CloudStorageAccount(account_name='pcslive', account_key=storage_key)
blob_service = account.create_block_blob_service()

year_ids = elb_repo.get_elb_harvest_year_ids(year=2016)

results = []
for idx, year_id in enumerate(year_ids):
    print("running elb prediction comparision.  idx, yearid: ({} of {}), {}".format(idx, len(year_ids), year_id))
    crop = gis_repo.get_pps_crop(year_id)
    if not 'Corn' in crop:
        print("found not-corn crop, ignoring: {}".format(crop))
        continue

    # use the indexed layer to find PL cells that are part of the ELB(s)
    # indexed layer is source layer ID 19
    elb_source_layers = [
        b.name
        for b in list(blob_service.list_blobs('sourcelayers', str(year_id)))
        if any(x in b.name for x in ['_13_', '_14_', '_15_'])]
    elb_harvest_source_layer_name = elb_source_layers[0] if len(elb_source_layers) > 0 else None

    if elb_harvest_source_layer_name is None:
        print("ELB has no indexed layer: {}".format(year_id))
        continue

    blob_zip = blob_service.get_blob_to_bytes('sourcelayers', elb_harvest_source_layer_name)

    vsiz = '/vsimem/{}.zip'.format(uuid.uuid4().hex)  # gdal/ogr requires a .zip extension
    FileFromMemBuffer(vsiz, bytes(blob_zip.content))
    with fiona.Collection(vsiz, vsi='zip') as f:
        shp = GeoDataFrame.from_features(f, crs={'init': 'epsg:4326'})

    elb_points = GeoDataFrame(shp.loc[shp['ELB_ID'] > 0])
    elb_centroids = list(elb_points.centroid)

    pps = gis_repo.processed_layer_shapes_by_year_id(year_id)
    # get pps cells that have an elb
    pps_elb_cells = pandas.DataFrame(
        pps.loc[pps['geometry'].apply(lambda x: any(x.intersects(c) for c in elb_centroids))])
    pps_elb_cells.drop(['geometry'], inplace=True, axis=1)

    # load weather record
    wx = gis_repo.weather_by_year_id(year_id)
    pps_elb_cells = pandas.concat([
        pps_elb_cells,
        pandas.DataFrame([wx.values], index=pps_elb_cells.index, columns=wx.keys())], axis=1)

    pps_elb_cells = pcs_data_loader.shape_pps_data(pps_elb_cells)
    pps_elb_cols = set(pps_elb_cells.columns.tolist())

    # add missing elb columns needed for the model
    missing_cols = train_cols - pps_elb_cols
    pps_elb_cells: pandas.DataFrame = pandas.concat(
        [pps_elb_cells,
         pandas.DataFrame(0, index=pps_elb_cells.index, columns=missing_cols)], axis=1
    )

    # remove any extra enum dummy columns in elb (that training isn't aware of)
    elb_extra_cols = set(pps_elb_cells.columns) - train_cols
    if any(elb_extra_cols):
        print(f"WARNING: ELB has unknown training enum (dummy) cols: {','.join(elb_extra_cols)}")
        pps_elb_cells.drop(elb_extra_cols, axis=1, inplace=True)

    elb_y = pps_elb_cells['Dry_Yield']
    elb_X = pps_elb_cells.drop(['Dry_Yield', 'Area'], axis=1)
    # order columns to match training
    elb_X = elb_X[X.columns]
    elb_score = score_util.score(extra_trees, scaler, elb_X, elb_y)
    print(elb_score)

    results.append((year_id, elb_score, elb_extra_cols))

rdf: pandas.DataFrame = pandas.concat(
    [
        pandas.DataFrame([_id for (_id, _, _) in results], columns=['year_id']),
        score_util.create_data_frame([scr for (_, scr, _) in results]),
        pandas.DataFrame(pandas.Series([c for (_, _, c) in results], name='extra_cols'))
    ],
    axis=1
)

rdf.to_csv('./results/20170823_elb_predictions/elb_harvest_predictions_results_{:%Y%m%d%H%m}.csv'.format(datetime.now()))
with open(f'./results/20170823_elb_predictions/elb_harvest_predictions_results_{datetime.now():%Y%m%d%H%m}.pickle', 'wb') as f:
    pickle.dump(results, f)
