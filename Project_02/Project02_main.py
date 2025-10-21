# ================================================================
# Project 02 Main Script
# ================================================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

workspace_root = Path(__file__).resolve().parent.parent   
sys.path.insert(0, str(workspace_root))

import functions as fn  

# ------------------------------------------------------------
# user settings
# ------------------------------------------------------------

# data file path
data_path = r"D:\One_Drive_Sajjad\OneDrive - Washington State University (email.wsu.edu)\Documents\Sajjad_Uddin_Mahmud\Courses\5. Fall 2025\MATH_564\Projects\WSU_Math564_Fall2025\Project_02\happiness.csv"

# data preparation settings
ward_col = "Ward/Neighborhood"
target_col = "Happiness"
feature_cols = [
    "Beauty of Neighborhood",
    "Convenience of Getting Around",
    "Housing Condition",
    "Street and Sidewalk Maintenance",
    "Public Schools",
    "Police Department",
    "Community Events",
    "City Services Information",
]
drop_row_missing_threshold = 6 # rows with >= this many missing features will be dropped
keep_raw_features = True  # set to False to overwrite original columns

# train/test split settings
test_size = 0.30
random_state = 42
stratify = True
strat_col = target_col

# ------------------------------------------------------------
# data preparation function
# ------------------------------------------------------------

def prepare_happiness_data(
    df,
    ward_col = "Ward/Neighborhood",
    target_col = "Happiness",
    feature_cols = None,
    drop_row_missing_threshold = 6,
    keep_raw_features = True,
):
    """
    prepare the somerville happiness dataset for modeling

    inputs
        df                          pandas dataframe with raw data
        ward_col                    column name for ward / neighborhood
        target_col                  column name for happiness (1..5)
        feature_cols                list of the eight satisfaction feature column names
        drop_row_missing_threshold  rows with >= this many missing features will be dropped
        keep_raw_features           if true, original 1..5 features are kept; scaled features
                                    are added with suffix '_scaled'. if false, originals are
                                    overwritten by scaled values.

    outputs
        df_out          processed dataframe
        prep_info       dictionary with imputation and drop statistics for reporting
    """

    if feature_cols is None:
        # set the expected eight feature names here
        feature_cols = [
            "Beauty of Neighborhood",
            "Convenience of Getting Around",
            "Housing Condition",
            "Street and Sidewalk Maintenance",
            "Public Schools",
            "Police Department",
            "Community Events",
            "City Services Information",
        ]

    df = df.copy()

    # ------------------------------------------------------------
    # drop unusable rows
    # ------------------------------------------------------------

    # remove rows with missing target
    n0 = len(df)
    mask_missing_target = df[target_col].isna()
    n_missing_target = int(mask_missing_target.sum())
    df = df.loc[~mask_missing_target].reset_index(drop = True)

    # remove rows with too many missing features
    feature_missing_count = df[feature_cols].isna().sum(axis = 1)
    mask_too_many_missing = feature_missing_count >= drop_row_missing_threshold
    n_drop_too_many_missing = int(mask_too_many_missing.sum())
    if n_drop_too_many_missing > 0:
        df = df.loc[~mask_too_many_missing].reset_index(drop = True)

    # ------------------------------------------------------------
    # compute medians (ward-level and global)
    # ------------------------------------------------------------

    # ward-level medians 
    ward_medians = {}
    for col in feature_cols:
        ward_medians[col] = df.groupby(ward_col)[col].median()

    # global medians
    global_medians = {col: float(df[col].median()) for col in feature_cols}

    # ------------------------------------------------------------
    # impute feature missing values by ward -> global
    # ------------------------------------------------------------

    # try ward-level median first, then global median
    for col in feature_cols:
        ward_median_col = df.groupby(ward_col)[col].transform("median")
        df[col] = df[col].fillna(ward_median_col)

        # fallback to global median if ward median was nan or still missing
        df[col] = df[col].fillna(global_medians[col])

    # ------------------------------------------------------------
    # normalize features to [0, 1] by (x - 1) / 4
    # ------------------------------------------------------------

    # scale function with clamping
    def scale_to_unit_interval(x):
        x = x.astype(float)

        # clamp to [1, 5] for safety, then scale
        x = np.clip(x, 1.0, 5.0)

        return (x - 1.0) / 4.0

    # apply scaling
    if keep_raw_features:
        for col in feature_cols:
            df[f"{col}_scaled"] = scale_to_unit_interval(df[col])
        scaled_feature_cols = [f"{c}_scaled" for c in feature_cols]
    else:
        for col in feature_cols:
            df[col] = scale_to_unit_interval(df[col])
        scaled_feature_cols = feature_cols

    # ------------------------------------------------------------
    # encode target as fractions {1/6, ..., 5/6}
    # ------------------------------------------------------------
    # happiness expected in {1, 2, 3, 4, 5}
    # encode as k / 6 where k in {1..5}
    df["Happiness_Encoded"] = df[target_col].astype(float) / 6.0

    # assemble prep info for reporting
    df_out = df
    prep_info = {
        "n_rows_initial" : n0,
        "n_missing_target_dropped" : n_missing_target,
        "n_drop_too_many_missing" : n_drop_too_many_missing,
        "feature_cols" : feature_cols,
        "scaled_feature_cols" : scaled_feature_cols,
        "ward_medians_available_for_cols" : {
            col: int(ward_medians[col].notna().sum()) for col in feature_cols
        },
        "global_medians" : global_medians,
        "scaling" : "(x - 1) / 4 with clamp to [1, 5]",
        "target_encoding" : "Happiness_Encoded = Happiness / 6",
        "ward_col" : ward_col,
        "target_col" : target_col,
    }

    return df_out, prep_info


if __name__ == "__main__":


    # ------------------------------------------------------------
    # load raw data
    # ------------------------------------------------------------
    data_path_obj = Path(data_path)
    df_raw = pd.read_csv(data_path_obj)

    # ------------------------------------------------------------
    # prepare data (no saving inside the function)
    # ------------------------------------------------------------
    df_proc, prep_info = prepare_happiness_data(
        df_raw,
        ward_col = ward_col,
        target_col = target_col,
        feature_cols = feature_cols,
        drop_row_missing_threshold = drop_row_missing_threshold,
        keep_raw_features = keep_raw_features,
    )

    # ------------------------------------------------------------
    # save processed data as csv (done here, not in the function)
    # ------------------------------------------------------------
    out_path = data_path_obj.with_name("happiness_processed.csv")
    df_proc.to_csv(out_path, index = False)

    # simple console report
    print("processed data saved to :", out_path)
    print("rows initial            :", prep_info["n_rows_initial"])
    print("rows dropped (target)   :", prep_info["n_missing_target_dropped"])
    print("rows dropped (>= missing threshold) :", prep_info["n_drop_too_many_missing"])

    # ------------------------------------------------------------
    # train/test split
    # ------------------------------------------------------------
    if stratify:
        df_train, df_test = fn.stratified_split_df(
            df_proc,
            target_col = strat_col,
            test_size = test_size,
            random_state = random_state,
    )
    else:
        # simple random split without stratification
        df_shuf = df_proc.sample(frac = 1.0, random_state = random_state).reset_index(drop = True)
        n_total = len(df_shuf)
        n_test  = int(round(test_size * n_total))
        n_test  = max(1, min(n_test, n_total - 1))
        df_test  = df_shuf.iloc[:n_test].reset_index(drop = True)
        df_train = df_shuf.iloc[n_test:].reset_index(drop = True)

    # simple train/test split report
    print("train rows :", len(df_train))
    print("test rows  :", len(df_test))
    print("train class counts :")
    print(df_train[strat_col].value_counts().sort_index())
    print("test class counts :")
    print(df_test[strat_col].value_counts().sort_index())

    # select features (scaled) and target (encoded)
    feat_cols = prep_info["scaled_feature_cols"]     
    y_col_enc = "Happiness_Encoded"                  

    X_train = df_train[feat_cols].to_numpy(dtype = float)
    y_train = df_train[y_col_enc].to_numpy(dtype = float).reshape(-1, 1)
    X_test  = df_test[feat_cols].to_numpy(dtype = float)
    y_test  = df_test[y_col_enc].to_numpy(dtype = float).reshape(-1, 1)

    # layer sizes for the required architecture
    layer_sizes = [X_train.shape[1], 12, 10, 1]

    # create feedforward neural network objective
    f, g = fn.make_ffnn_objective(X_train, y_train, layer_sizes = layer_sizes, l2 = 0.0)

    # initialize parameters
    w0 =  fn.ffnn_init_params(layer_sizes, random_state = np.random.default_rng(42))

    # ------------------------------------------------------------
    # helper: convert scalar outputs in (0,1) to classes {1..5}
    # targets are spaced at {1/6, 2/6, 3/6, 4/6, 5/6}
    # ------------------------------------------------------------
    targets_enc = np.arange(1, 6, dtype = float).reshape(-1, 1) / 6.0  # shape (5,1)
    targets_cls = np.arange(1, 6, dtype = int).reshape(-1, 1)          # shape (5,1)

    def decode_to_class(yhat):
        # yhat: (n,1) in (0,1); snap to nearest of the five target points
        diffs = np.abs(yhat - targets_enc.T)   # (n,5)
        idx   = np.argmin(diffs, axis = 1)     # (n,)
        return targets_cls[idx, 0]             # (n,)
    
    
    # ------------------------------------------------------------
    # run method 1: bfgs + strong wolfe 
    # ------------------------------------------------------------
    bfgs_opts = {"max_iter" : 500, "tol" : 1e-6}
    res_bfgs = fn.quasi_newton_bfgs(
        f = f,
        grad = g,
        x0 = w0,
        line_search = fn.strong_wolfe,
        opts = bfgs_opts
    )
    w_bfgs = res_bfgs["x"]

    # ------------------------------------------------------------
    # run method 2: gradient descent + strong wolfe 
    # ------------------------------------------------------------
    from functions import gradient_descent

    gd_opts = {"max_iter" : 200, "tol" : 1e-6}
    res_gd = fn.gradient_descent(
        f = f,
        grad = g,
        x0 = w0,
        line_search = fn.strong_wolfe,
        opts = gd_opts
    )
    w_gd = res_gd["x"]



    # ------------------------------------------------------------
    # prediction helper
    # ------------------------------------------------------------
    def ffnn_predict(X, w, layer_sizes):
        shapes = []
        sizes = []
        for l in range(1, len(layer_sizes)):
            d_prev = layer_sizes[l - 1]
            d_curr = layer_sizes[l]
            shapes.append(((d_prev, d_curr), (d_curr,)))
            sizes.append((d_prev * d_curr, d_curr))
        idx = 0
        slices = []
        for (w_count, b_count) in sizes:
            sW = slice(idx, idx + w_count)
            idx += w_count
            sb = slice(idx, idx + b_count)
            idx += b_count
            slices.append((sW, sb))

        def sigmoid(u):
            return 1.0 / (1.0 + np.exp(-u))

        a = X
        for (sW, sb), (shapeW, shapeb) in zip(slices, shapes):
            W = w[sW].reshape(shapeW)
            b = w[sb].reshape(shapeb)
            a = sigmoid(a @ W + b)
        return a  
    
    # test predictions for both methods
    yhat_bfgs = ffnn_predict(X_test, w_bfgs, layer_sizes)
    yhat_gd   = ffnn_predict(X_test, w_gd,   layer_sizes)

    print("GD yhat min/max/mean:", float(yhat_gd.min()), float(yhat_gd.max()), float(yhat_gd.mean()))
    print("Unique predicted classes (GD):", np.unique(decode_to_class(yhat_gd)))
    print("Are GD weights unchanged from init?:", np.allclose(w0, w_gd))

    ycls_bfgs = decode_to_class(yhat_bfgs)
    ycls_gd   = decode_to_class(yhat_gd)
    ytrue_cls = decode_to_class(y_test)

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm_bfgs = pd.crosstab(
        pd.Series(ytrue_cls.flatten(), name = "True"),
        pd.Series(ycls_bfgs.flatten(), name = "Predicted"),
    )
    cm_gd = pd.crosstab(
        pd.Series(ytrue_cls.flatten(), name = "True"),
        pd.Series(ycls_gd.flatten(), name = "Predicted"),
    )

    print("\nConfusion Matrix: BFGS + Strong Wolfe")
    print(cm_bfgs)
    print("\nConfusion Matrix: Gradient Descent + Strong Wolfe")
    print(cm_gd)

    classes = [1, 2, 3, 4, 5]

    # reindex to full 1...5 grid for neat printing/saving
    cm_bfgs = cm_bfgs.reindex(index = classes, columns = classes, fill_value = 0)
    cm_gd   = cm_gd.reindex(index = classes, columns = classes, fill_value = 0)


    # ------------------------------------------------------------
    # save to ./results/evaluation.txt
    # ------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents = True, exist_ok = True)
    with open(os.path.join(results_dir, "evaluation.txt"), "w") as f:
        f.write("confusion matrix: bfgs + strong wolfe\n")
        f.write(str(cm_bfgs) + "\n\n")
        f.write("confusion matrix: gradient descent + strong wolfe\n")

    print("saved confusion matrices to ./results/evaluation.txt")

# ================================================================