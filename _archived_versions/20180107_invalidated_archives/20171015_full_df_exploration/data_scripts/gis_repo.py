import pandas
import geopandas
from xml.etree import ElementTree
from shapely.geometry import Polygon

from sqlalchemy import (
    text as sql_text,
    create_engine
)

import util.config

settings = util.config.load_settings()
gis_db = create_engine(settings['db']['gis'])


def pps_cols():
    """
    Gets ML pps columns.

    :return: list tuple result: ('colname', 'datatype')
    """

    sql_ = """
        SELECT
          COLUMN_NAME AS Name, DATA_TYPE AS DataType
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'PremierProcessedShapes'
            AND column_name NOT LIKE '%$%'
            AND column_name NOT LIKE '%cost'
            AND column_name NOT IN (
              'uid', 'shapeuid', 'ShapeIndex', 'ShapeX', 'ShapeXml', 'ShapeY',
              'createdate', 'lastupdated', 'mgmtzone', 'plntdate', 'harvdate', 'gridcolumn', 'gridrow')
        ORDER BY COLUMN_NAME;"""

    return list(gis_db.execute(sql_))


def processed_layer_shapes_by_year_id(year_id: int, *extra_cols: str) -> geopandas.GeoDataFrame:
    """
    Loads a GIS Processed Layer into a GeoDataFrame

    :param year_id:
    :param extra_cols:
    :return: geo dataframe of the pl pps, with shapexml converted to shapely geometry
    """
    col_names = [name for (name, dtype) in pps_cols()]
    extra_cols = extra_cols or []

    cols = set(list(extra_cols) + col_names)
    cols.add('ShapeXml')

    cols_sql = ','.join(cols)

    sql_ = sql_text(f"""
        SELECT {cols_sql}
        FROM GIS..PremierProcessedShapes pps with (NOLOCK)
        join gis..ProcessedLayer pl with (NOLOCK) on pl.UID = pps.ProcessedLayerUID
        join PlatformManager..YearIDMapping ym on pl.HierarchyItemUID = ym.YearUID
        where ym.ID = :YearId
    """)

    df = pandas.read_sql(sql_, con=gis_db, params={'YearId': year_id})

    shape_xml_polygons = [parse_shape_xml(xml) for xml in df['ShapeXml']]
    df.drop('ShapeXml', axis=1, inplace=True)

    crs_ = {'init': 'epsg:4326'}
    geo_df = geopandas.GeoDataFrame(df, crs=crs_, geometry=shape_xml_polygons)
    return geo_df


def weather_by_year_id(year_id: int) -> pandas.Series:
    sql = """
SELECT TOP 1 w.*
FROM gis..PCS_Weather w
  join PlatformManager..YearIDMapping ym on ym.YearUID = w.YearUID
WHERE ym.ID = :year_id"""

    exclude_cols = ['FieldID', 'YearUID']
    df = pandas.read_sql(sql_text(sql), con=gis_db, params={'year_id': year_id})
    df.drop(exclude_cols, axis=1, inplace=True)

    return df.iloc[0]


def get_pps_crop(year_id: int) -> str:
    sql = """
SELECT top 1 pps.Crop
FROM GIS..PremierProcessedShapes as pps with (NOLOCK)
  join gis..ProcessedLayer as pl with (NOLOCK ) on pl.UID = pps.ProcessedLayerUID
  join PlatformManager..YearIDMapping ym on ym.YearUID = pl.HierarchyItemUID
where ym.id = :year_id;"""

    return gis_db.execute(sql_text(sql), year_id=year_id).scalar()


def parse_shape_xml(xml: str) -> Polygon:
    """
    Parses GIS.PremierProcessedShape.ShapeXml into a shapely polygon
    :param xml:
    :return:
    """

    xtree = ElementTree.fromstring(xml)
    rings = xtree.findall('Rings')[0]

    parsed_rings = []
    for idx, r in enumerate(rings):
        parsed = []
        for point in r[0]:
            parsed.append((float(point[0].text), float(point[1].text)))

        parsed_rings.append(parsed)

        poly = Polygon(parsed_rings[0], parsed_rings[1:])
        return poly
