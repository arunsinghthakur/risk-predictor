from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=remove_outliers,
                inputs=["house_prices", "parameters"],
                outputs="house_prices_no_outliers",
                name="outliers_node",
            ),
            node(
                func=fill_na,
                inputs=["house_prices_no_outliers", "parameters"],
                outputs="house_prices_no_na",
                name="fill_na_node",
            ),
            node(
                func=total_sf,
                inputs="house_prices_no_na",
                outputs="house_prices_sf",
                name="total_sf_node",
            ),
            node(
                func=house_prices_clean,
                inputs=["house_prices_sf", "parameters"],
                outputs="model_input_table",
                name="house_prices_clean_node",
            ),
        ]
    )