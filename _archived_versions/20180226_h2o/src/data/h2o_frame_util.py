from typing import List

import pickle
from h2o import H2OFrame, h2o


def grouped_kfold(frame: H2OFrame, n_folds: int, src_col_name: str, dest_col_name='_kfold', seed=-1, remove_frame=True):
    src_col_frame = frame[src_col_name]
    group_col_uniq = src_col_frame.unique()

    print(f"kfold group unique val count:: {src_col_name}, {group_col_uniq.nrows}")
    kfold_col_frame = group_col_uniq.kfold_column(n_folds, seed)
    group_col_kfold = group_col_uniq.cbind(kfold_col_frame)
    group_col_kfold_named = group_col_kfold.set_names([src_col_name, dest_col_name])
    kfold_frame = frame.merge(group_col_kfold_named)
    # force eval...
    print(f"merged frame id: {kfold_frame.frame_id}")

    remove_frames([src_col_frame, group_col_uniq, group_col_kfold, kfold_col_frame, group_col_kfold_named])

    if remove_frame:
        h2o.remove(frame.frame_id)
    return kfold_frame


def remove_frames(frames: List[H2OFrame]):
    for f in frames:
        h2o.remove(f.frame_id)


def load_gis_pps(df_csv_path: str, col_types_path: str) -> H2OFrame:
    print(f"loading col types: {col_types_path}")
    with open(col_types_path, 'rb') as f:
        col_types = pickle.load(f)

    print(f"loading csv into h2o: {df_csv_path}")
    df = h2o.upload_file(df_csv_path)

    print(f"ensuring column types")
    for c in col_types:
        c_type = col_types[c]
        prev_frame_id = df.frame_id
        if c_type == 'numeric' and not df[c].isnumeric()[0]:
            print(f"ensuring numeric: {c}")
            df[c] = df[c].asnumeric()
        elif c_type == 'enum' and not df[c].isfactor()[0]:
            print(f"ensuring enum: {c}, currently: {df.type(c)}")
            df[c] = df[c].ascharacter().asfactor()

        if prev_frame_id != df.frame_id:
            h2o.remove(prev_frame_id)

    # make year ID numeric
    if not df['YearID'].isnumeric()[0]:
        print(f"converting year ID back to numeric from: {df.type('YearID')}")
        prev_frame_id = df.frame_id
        df['YearID'] = df['YearID'].asnumeric()
        # force eval...
        print(f"converted, new frame id: {df.frame_id}")
        h2o.remove(prev_frame_id)

    return df


def remove_outliers(
        data: h2o.H2OFrame,
        quantile_min=.002, quantile_max=.998,
        max_targ_pop=80_000) -> H2OFrame:
    # remove outliers
    # explicitly remove targ pop > X
    cols = set(data.columns_by_type(coltype='numeric'))
    cols = sorted(list(cols - set(exclude_columns + yield_dep_columns)))

    rows = data.shape[0]
    data = data[data['TargPop'] <= max_targ_pop]
    quantile_min = data[cols].quantile(quantile_min)
    quantile_max = data[cols].quantile(quantile_max)

    for c in cols:
        min_val = quantile_min[c]
        max_val = quantile_max[c]

        print(f"LOG removing outliers: {c} >>> min: {min_val}, max: {max_val} ")

        if np.any(np.isnan([min_val, max_val])):
            print("LOG min/max is NaN, skipping")
            continue

        rows_prev_ = data.shape[0]
        data = data[(data[c] >= min_val) & (data[c] <= max_val)]
        print(f"dropped: {rows_prev_ - data.shape[0]}")

    dropped_count = rows - data.shape[0]
    print(f"dropped outlier rows: {dropped_count}, {(dropped_count/rows) * 100:.1f}%")

    if reindex:
        print(f"reindexing after dropped rows")
        data.reset_index(drop=True, inplace=True)

    return data

