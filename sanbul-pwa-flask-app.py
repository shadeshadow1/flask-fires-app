import numpy as np
import pandas as pd

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
np.random.seed(42)

app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"

# --------------------------------------------------
# 입력 폼
# --------------------------------------------------
class LabForm(FlaskForm):
    longitude = StringField("longitude(1-7)", validators=[DataRequired()])
    latitude = StringField("latitude(1-7)", validators=[DataRequired()])
    month = StringField("month(01-Jan ~ Dec-12)", validators=[DataRequired()])
    day = StringField("day(00-sun ~ 06-sat, 07-hol)", validators=[DataRequired()])
    avg_temp = StringField("avg_temp", validators=[DataRequired()])
    max_temp = StringField("max_temp", validators=[DataRequired()])
    max_wind_speed = StringField("max_wind_speed", validators=[DataRequired()])
    avg_wind = StringField("avg_wind", validators=[DataRequired()])
    submit = SubmitField("Submit")

# --------------------------------------------------
# 1) 데이터 불러오기
# --------------------------------------------------
fires = pd.read_csv("sanbul2district-divby100.csv")

# burned_area 로그 변환
fires["burned_area"] = np.log1p(fires["burned_area"])

# 입력(X), 정답(y) 분리
X = fires.drop("burned_area", axis=1)
y = fires["burned_area"].copy()

# --------------------------------------------------
# 2) 전처리 파이프라인 만들기
# --------------------------------------------------
num_attribs = [
    "longitude", "latitude", "avg_temp",
    "max_temp", "max_wind_speed", "avg_wind"
]
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ("std_scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

X_prepared = full_pipeline.fit_transform(X)

# --------------------------------------------------
# 3) 모델 학습하기
# --------------------------------------------------
model = LinearRegression()
model.fit(X_prepared, y)

# --------------------------------------------------
# 라우트
# --------------------------------------------------
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def lab():
    form = LabForm()

    if form.validate_on_submit():
        user_data = pd.DataFrame([{
            "longitude": float(form.longitude.data),
            "latitude": float(form.latitude.data),
            "month": form.month.data.strip(),
            "day": form.day.data.strip(),
            "avg_temp": float(form.avg_temp.data),
            "max_temp": float(form.max_temp.data),
            "max_wind_speed": float(form.max_wind_speed.data),
            "avg_wind": float(form.avg_wind.data),
        }])

        user_prepared = full_pipeline.transform(user_data)
        pred_log = model.predict(user_prepared)[0]

        # 로그 변환 복원
        prediction = np.expm1(pred_log)

        # 음수 방지
        prediction = max(0, prediction)

        return render_template("result.html", prediction=round(prediction, 2))

    return render_template("prediction.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)