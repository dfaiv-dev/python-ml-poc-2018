from sqlalchemy import create_engine, text as sql_text

import util.config

settings = util.config.load_settings()
field_info_db = create_engine(settings['db']['fieldinfo'])


def get_enroll_year_id(enroll_id: int):
    sql = """    
        SELECT ym.ID AS YearId
        FROM FieldInfo.trials.TrialEnrollment te
          JOIN PlatformManager..YearIDMapping ym ON ym.YearUID = te.YearUID
        WHERE te.id = :enroll_id"""

    return field_info_db.execute(sql_text(sql), enroll_id=enroll_id).scalar()


def get_elb_harvest_year_ids(year=2016):
    sql = """
        SELECT DISTINCT (l.yearid) AS yearid
        FROM Wolverine.layers.SourceLayerDataType dt
          JOIN Wolverine.layers.SourceLayer AS l ON l.SourceLayerDataTypeID = dt.ID
          JOIN PlatformManager..YearIDMapping ym ON ym.id = l.YearID
        WHERE dt.name LIKE '%elb harvest%' AND ym.Year = :year
        ORDER BY yearid"""

    return [r for (r,) in list(
        field_info_db.execute(sql_text(sql), year=year)
    )]
