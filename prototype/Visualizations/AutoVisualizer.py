import pandas as pd
import plotly.express as px
from typing import List, Tuple



def to_dataframe(rows: List[Tuple], columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)

def plot_univariate(df: pd.DataFrame, column: str):
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df, 
            x=column, 
            marginal="box", 
            nbins=30, 
            title=f"Distribution of {column}"
        )
    else:
        vc = df[column].value_counts().reset_index()
        vc.columns = [column, "count"]  # Rename properly for Plotly
        fig = px.bar(
            vc, 
            x=column, 
            y="count", 
            labels={column: column, "count": "Count"}, 
            title=f"Frequency of {column}"
        )
    return fig


def plot_bivariate(df: pd.DataFrame, x: str, y: str):
    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
    elif pd.api.types.is_numeric_dtype(df[y]):
        fig = px.box(df, x=x, y=y, title=f"{y} distribution across {x}")
    else:
        fig = px.histogram(df, x=x, color=y, barmode='group', title=f"{x} vs {y}")
    return fig


def plot_multivariate(df: pd.DataFrame, dimensions: List[str]):
    if len(dimensions) == 3:
        fig = px.scatter_3d(df, x=dimensions[0], y=dimensions[1], z=dimensions[2],
                            color=dimensions[0], title=f"3D Plot: {dimensions}")
    elif len(dimensions) > 3:
        fig = px.parallel_coordinates(df, dimensions=dimensions, title="Parallel Coordinates Plot (ND)")
    else:
        print("Need at least 3 columns for multivariate plot")
        return
    return fig


def auto_visualize(df: pd.DataFrame):
    figs = []
    numeric = df.select_dtypes(include='number').columns.tolist()
    if len(numeric) == 1:
        figs.append(plot_univariate(df, numeric[0]))
    elif len(numeric) == 2:
        figs.append(plot_bivariate(df, numeric[0], numeric[1]))
    elif len(numeric) >= 3:
        figs.append(plot_multivariate(df, numeric[:4]))  # choose first 4 numeric cols
    else:
        # Fallback to categorical
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                figs.append(plot_univariate(df, col))
                break
    return figs




