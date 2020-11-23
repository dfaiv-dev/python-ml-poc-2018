import logging
import pandas
import re
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

pcsMLDbConnStr = 'mssql+pyodbc://localhost/PcsML?driver=SQL+Server+Native+Client+11.0'
engine = create_engine(pcsMLDbConnStr)

group_col_sql = """
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'PpsGrouping'
  AND (DATA_TYPE = 'varchar' OR COLUMN_NAME IN ('PlntDate#', 'HarvDate#'));
"""

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

group_cols = list(map(lambda r: r[0], engine.execute(group_col_sql)))


def load_corn_data_frame():
    df = pandas.read_sql_query(data_query_sql, pcsMLDbConnStr)
    print("data shape: " + str(df.shape))

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
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(value=0, inplace=True)

    logger.info("DF final shape: %s" % str(df.shape))
    return df
