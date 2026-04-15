import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires = pd.read_csv("./sanbul-5.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] +1)


print(fires.head())
print(fires.info())
print(fires.describe())
print(fires["month"].value_counts())
print(fires["day"].value_counts())



# 수치형 컬럼만 선택
num_cols = fires.select_dtypes(include=["float64", "int64"]).columns

# 히스토그램 그리기
fires[num_cols].hist(figsize=(12, 8), bins=30)

plt.suptitle("Numerical Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

#1 -4

fires["log_burned_area"] = np.log1p(fires["burned_area"])

plt.subplot(1, 2, 2)
plt.hist(fires["log_burned_area"], bins=30)
plt.title("ln(burned_area + 1)")

plt.tight_layout()
plt.show()


# 1-5 

train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()

fires["month"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]


print("\nMonth category proportion: \n",
      strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion: \n",
      fires["month"].value_counts() / len(fires))

#1-6
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 사용할 컬럼 선택 (4개 이상)
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]

scatter_matrix(fires[attributes], figsize=(10, 8))

plt.show()


# 1-7
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=fires["max_temp"], label="max_temp", c = "burned_area", cmap=plt.get_cmap("jet"), colorbar=True)


#1 -8

from sklearn.preprocessing import OneHotEncoder

fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)

fires["month"] = fires["month"].str.strip()
fires["day"] = fires["day"].str.strip()

cat_month_encoder = OneHotEncoder()
fires_month_1hot = cat_month_encoder.fit_transform(fires[["month"]])

cat_day_encoder = OneHotEncoder()
fires_day_1hot = cat_day_encoder.fit_transform(fires[["day"]])

print("month 인코딩 결과:")
print(fires_month_1hot)
print(cat_month_encoder.categories_)

print("\nday 인코딩 결과:")
print(fires_day_1hot)
print(cat_day_encoder.categories_)


print("\n###############################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 수치형 / 카테고리형 분리
num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

# 수치형 파이프라인
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# 전체 파이프라인
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

# 변환 실행
fires_prepared = full_pipeline.fit_transform(fires)

print(fires_prepared.shape)