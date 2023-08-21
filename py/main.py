import pathlib

import pandas as pd
import sklearn.metrics as sklmetrics
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def format_data_cols(data: pd.DataFrame) -> None:
    """For formatting the columns for the input frame.

    :param data:
    :return:
    """
    # dropping columns that are irrelevant or annoying or too informative or correlate with others
    cols_to_drop = [
        "next_pymnt_d",
        "last_pymnt_d",
        "last_credit_pull_d",
        "loan_status",
        "issue_d",
        "earliest_cr_line",
        "member_id",
        "zip_code",
        "addr_state",
        "funded_amnt",
        "funded_amnt_inv",
        "installment",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "last_pymnt_amnt",
    ]
    for col_name in data.columns:
        if col_name.startswith("Unnamed: "):
            print(f"Dropping {col_name}...")
            cols_to_drop.append(col_name)

    data.drop(labels=cols_to_drop, axis=1, inplace=True)
    # filling columns
    data.fillna(
        value={"mths_since_last_delinq": 0, "emp_length": "0", "annual_inc": 0, "revol_bal": 0, "revol_util": "0"},
        inplace=True,
    )

    data["term"] = data["term"].str.slice(stop=-7).astype(int)

    data["revol_util"] = data["revol_util"].str.removesuffix("%").astype(float)

    data["emp_length"] = (
        data["emp_length"]
        .str.removeprefix("< ")
        .str.removesuffix("s")
        .str.removesuffix(" year")
        .str.removesuffix("+")
        .astype(int)
    )

    data["repay_fail"] = data["repay_fail"] == 1


def replace_nominal_data(data: pd.DataFrame) -> pd.DataFrame:
    """For replacing the nominal columns in data with dummy columns

    :param data:
    :return:
    """
    nominal_col_names = ["home_ownership", "verification_status", "purpose"]
    dummy_df = pd.get_dummies(data[nominal_col_names], drop_first=True)
    replaced = pd.concat(objs=[data, dummy_df], axis=1).drop(nominal_col_names, axis=1)
    return replaced


def get_numeric_column_names(data: pd.DataFrame) -> list:
    """For getting a list of all the names of the numeric columns in data

    :param data:
    :return:
    """
    return [
        col_name
        for col_name, data_type in data.dtypes.items()
        if pd.api.types.is_numeric_dtype(data_type) and data_type != bool
    ]


proj_dir = pathlib.Path("/home/thomas/classifier_practice")
csv_dir = proj_dir / "csv"

data = pd.read_csv(csv_dir / "Anonymize_Loan_Default_data.csv", encoding="ISO-8859-1", skiprows=[1, 55], index_col=1)
format_data_cols(data=data)
logistic_data = replace_nominal_data(data=data)
num_col_names = get_numeric_column_names(data=logistic_data)
sc = StandardScaler()
logistic_data[num_col_names] = sc.fit_transform(logistic_data[num_col_names])

target_name = "repay_fail"
x_val = logistic_data.drop(labels=[target_name], axis=1)
y_val = logistic_data[target_name]

estimator = LogisticRegression(class_weight="balanced", solver="liblinear")
rfe = RFE(estimator=estimator, n_features_to_select=20).fit(x_val, y_val)

x_chosen = x_val.columns[rfe.support_]
X = x_val[x_chosen]

train_X, test_X, train_y, test_y = train_test_split(X, y_val, train_size=0.8, random_state=1)
print("Train Features: ", train_X.shape, "Test Features: ", test_X.shape)
print("Train Labels: ", train_y.shape, "Test Labels: ", test_y.shape)

n_sample = train_y.shape[0]
n_pos_sample = train_y[train_y == 0].shape[0]
n_neg_sample = train_y[train_y == 1].shape[0]
print(
    "Observations: {}; Positives: {:.2%}; Negatives: {:.2%}".format(
        n_sample, n_pos_sample / n_sample, n_neg_sample / n_sample
    )
)
print("Features: ", train_X.shape[1])

sm = SMOTE(random_state=1)
train_X, train_y = sm.fit_resample(train_X, train_y)
print("After SMOTE: ")
n_sample = train_y.shape[0]
n_pos_sample = train_y[train_y == 0].shape[0]
n_neg_sample = train_y[train_y == 1].shape[0]
print(
    "Observations: {}; Positives: {:.2%}; Negatives: {:.2%}".format(
        n_sample, n_pos_sample / n_sample, n_neg_sample / n_sample
    )
)

model = LogisticRegression(solver="liblinear")
model.fit(train_X, train_y)

predict_y = model.predict(test_X)
sklmetrics.accuracy_score(test_y, predict_y)
