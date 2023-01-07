FROM quay.io/astronomer/ap-airflow:1.10.15-8-buster-onbuild
RUN pip install --user dist/risk_predictor-0.1-py3-none-any.whl