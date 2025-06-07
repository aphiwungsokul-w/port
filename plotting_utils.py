import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

__all__ = [
    "weights_bar",
    "efficient_frontier",
    "cov_heatmap",
]

# ──────────────────────────────────────────────────────────────
# 1) Weights bar – Plotly
# ──────────────────────────────────────────────────────────────

def weights_bar(weights: dict[str, float], *, title: str = "Portfolio Weights") -> go.Figure:
    """Return a horizontal bar chart of portfolio weights (Plotly)."""
    if not weights:
        return go.Figure()

    df = (
        pd.Series(weights, name="Weight")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Ticker"})
    )

    fig = px.bar(
        df,
        x="Weight",
        y="Ticker",
        orientation="h",
        text="Weight",
        title=title,
    )
    fig.update_layout(height=300, margin=dict(l=80, r=20, t=40, b=40))
    fig.update_traces(texttemplate="%{text:.2%}", hovertemplate="%{y}: %{x:.2%}<extra></extra>")
    fig.update_xaxes(tickformat="%")
    return fig


# ──────────────────────────────────────────────────────────────
# 2) Efficient‑Frontier (random sampling) – Plotly
# ──────────────────────────────────────────────────────────────

def _random_portfolios(n_assets: int, n_points: int, rng: np.random.Generator):
    """Generate *n_points* random long‑only weight vectors that sum to 1."""
    return rng.dirichlet(np.ones(n_assets), size=n_points)


def efficient_frontier(
    port,
    *,
    tickers: list[str],
    points: int = 300,
    rf: float = 0.02,
    show_assets: bool = True,
):
    """Return (Plotly Figure, DataFrame) of random‑sample Efficient Frontier.

    When *show_assets* = True (default) the function also overlays each
    **individual stock** σ–µ point so users can see that random points
    are within the bounds of the actual selected stocks.
    """
    
    # Debug print to see what we're working with
    print(f"Original tickers count: {len(tickers)}")
    print(f"Port.mu type: {type(port.mu)}")
    print(f"Port.cov type: {type(port.cov)}")
    
    # Handle different types of port.mu (Series vs DataFrame)
    if isinstance(port.mu, pd.DataFrame):
        mu_index = port.mu.columns.tolist()
    elif isinstance(port.mu, pd.Series):
        mu_index = port.mu.index.tolist()
    else:
        print("Warning: port.mu is neither DataFrame nor Series")
        return go.Figure(layout_title_text="Invalid port.mu format"), pd.DataFrame()
    
    # Handle port.cov index
    if hasattr(port.cov, 'index'):
        cov_index = port.cov.index.tolist()
    else:
        print("Warning: port.cov has no index")
        return go.Figure(layout_title_text="Invalid port.cov format"), pd.DataFrame()
    
    print(f"Available mu assets: {len(mu_index)}")
    print(f"Available cov assets: {len(cov_index)}")
    
    # Filter tickers to only include those present in both mu and cov
    valid_tickers = [t for t in tickers if t in mu_index and t in cov_index]
    print(f"Valid tickers after filtering: {len(valid_tickers)}")
    
    if len(valid_tickers) < 2:
        print("Not enough valid tickers for frontier")
        # Return a simple message figure
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Need ≥ 2 assets for frontier<br>Valid assets: {len(valid_tickers)}<br>Available: {valid_tickers[:5]}...",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Efficient Frontier - Insufficient Data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=450,
            width=600
        )
        return fig, pd.DataFrame()

    # Extract mu and cov for valid tickers
    try:
        if isinstance(port.mu, pd.DataFrame):
            mu = port.mu.loc[:, valid_tickers].iloc[0].to_numpy(dtype=float)
        else:
            mu = port.mu.loc[valid_tickers].to_numpy(dtype=float)
        
        cov = port.cov.loc[valid_tickers, valid_tickers].to_numpy(dtype=float)
        
        print(f"Mu shape: {mu.shape}")
        print(f"Cov shape: {cov.shape}")
        
    except Exception as e:
        print(f"Error extracting mu/cov: {e}")
        return go.Figure(layout_title_text=f"Data extraction error: {str(e)}"), pd.DataFrame()

    # Generate random portfolios
    rng = np.random.default_rng(42)
    n = len(valid_tickers)
    W = _random_portfolios(n, points, rng)

    # Calculate returns and risks
    rets = W @ mu
    sigs = np.sqrt(np.einsum("ij,ij->i", W @ cov, W))
    
    # Calculate Sharpe ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (rets - rf) / sigs

    # Create DataFrame
    df = pd.DataFrame({"Risk": sigs, "Return": rets, "Sharpe": sharpe})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        return go.Figure(layout_title_text="No valid portfolio points generated"), pd.DataFrame()

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Risk",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Viridis",
        hover_data={"Risk": ":.2%", "Return": ":.2%", "Sharpe": ":.2f"},
        title="Efficient Frontier (random sample)",
    )

    # Add individual assets if requested
    if show_assets:
        try:
            asset_risks = np.sqrt(np.diag(cov))
            asset_df = pd.DataFrame({
                "Risk": asset_risks,
                "Return": mu,
                "Ticker": valid_tickers,
            })
            
            fig.add_trace(
                go.Scatter(
                    x=asset_df["Risk"],
                    y=asset_df["Return"],
                    mode="markers",
                    marker=dict(symbol="diamond", size=10, color="red", line=dict(width=2, color="black")),
                    name="Individual Assets",
                    hovertemplate="<b>%{hovertext}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
                    hovertext=asset_df["Ticker"],
                )
            )
        except Exception as e:
            print(f"Error adding individual assets: {e}")

    fig.update_layout(
        height=450, 
        width=600, 
        margin=dict(l=60, r=20, t=50, b=50),
        xaxis_title="Risk (σ)",
        yaxis_title="Expected Return (μ)"
    )
    
    return fig, df[["Risk", "Return", "Sharpe"]]


# ──────────────────────────────────────────────────────────────
# 3) Correlation / Covariance Heat‑map – Plotly
# ──────────────────────────────────────────────────────────────

def cov_heatmap(cov_df: pd.DataFrame, *, corr: bool = True) -> go.Figure:
    """Return a correlation (default) or covariance heat‑map (Plotly)."""
    if cov_df.empty:
        return go.Figure(layout_title_text="No data for heatmap")
    
    try:
        mat = cov_df.corr() if corr else cov_df.copy()
        
        fig = px.imshow(
            mat,
            color_continuous_scale="RdBu",
            origin="lower",
            aspect="auto",
            title="Correlation Heat‑map" if corr else "Covariance Heat‑map",
            labels=dict(x="Assets", y="Assets", color="Correlation" if corr else "Covariance")
        )
        
        fig.update_layout(
            height=450, 
            width=450, 
            margin=dict(l=60, r=60, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return go.Figure(layout_title_text=f"Heatmap error: {str(e)}")