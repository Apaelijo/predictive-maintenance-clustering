import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dataset_metadata_overview(df: pd.DataFrame, figsize: tuple = (14, 5), show: bool = True):
    """
    Create a lightweight metadata overview for an EDA notebook.

    The plot highlights missing values, column dtypes, and a compact dataset summary
    without changing the underlying data or the existing analysis workflow.
    """
    if df is None:
        raise ValueError("A pandas DataFrame is required.")

    metadata = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "unique_values": df.nunique(dropna=False),
        }
    ).reset_index().rename(columns={"index": "column"})

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Dataset Metadata Overview", fontsize=14, fontweight="bold")

    missing = metadata.sort_values("missing_values", ascending=False).head(15)
    sns.barplot(
        data=missing,
        x="missing_values",
        y="column",
        color="steelblue",
        ax=axes[0],
    )
    axes[0].set_title("Missing values by column")
    axes[0].set_xlabel("Missing values")
    axes[0].set_ylabel("")

    dtype_counts = metadata["dtype"].value_counts().reset_index()
    dtype_counts.columns = ["dtype", "count"]
    sns.barplot(
        data=dtype_counts,
        x="count",
        y="dtype",
        color="mediumseagreen",
        ax=axes[1],
    )
    axes[1].set_title("Column types")
    axes[1].set_xlabel("Number of columns")
    axes[1].set_ylabel("")

    axes[2].axis("off")
    summary_text = (
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}\n"
        f"Duplicate rows: {df.duplicated().sum()}\n"
        f"Numeric columns: {df.select_dtypes(include='number').shape[1]}\n"
        f"Categorical columns: {df.select_dtypes(exclude='number').shape[1]}"
    )
    axes[2].text(
        0.05,
        0.95,
        summary_text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        wrap=True,
    )
    axes[2].set_title("Dataset summary")

    plt.tight_layout()
    if show:
        plt.show()

    return metadata, fig


def plot_variable_summary(df: pd.DataFrame, features=None, figsize: tuple = (16, 6), show: bool = True):
    """
    Create a presentation-friendly summary of selected variables.

    Numeric features are shown with histograms and categorical features with count plots,
    making it easier to communicate the dataset profile in a technical presentation.
    """
    if df is None:
        raise ValueError("A pandas DataFrame is required.")

    if features is None:
        features = list(df.columns)
    else:
        features = [feature for feature in features if feature in df.columns]

    if not features:
        raise ValueError("The DataFrame does not contain any variables to summarize.")

    numeric_features = [feature for feature in features if pd.api.types.is_numeric_dtype(df[feature])]
    categorical_features = [feature for feature in features if feature not in numeric_features]

    n_cols = min(3, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
    fig.suptitle("Variable Summary", fontsize=14, fontweight="bold")

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax, feature in zip(axes, features):
        if feature in numeric_features:
            sns.histplot(df[feature], kde=True, ax=ax, color="steelblue")
            ax.set_title(feature)
            ax.set_xlabel("")
            ax.set_ylabel("Count")
        else:
            sns.countplot(data=df, x=feature, ax=ax, color="mediumseagreen")
            ax.set_title(feature)
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for ax in axes[len(features):]:
        ax.remove()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()

    return fig


def plot_clustering_tendency_preview(df: pd.DataFrame, features=None, figsize: tuple = (8, 6), show: bool = True):
    """
    Show a lightweight 2D PCA projection as a visual sanity check for clusterability.

    This is a good companion to the Hopkins statistic explanation because it gives a quick
    view of whether the selected features appear to form separable groups.
    """
    if df is None:
        raise ValueError("A pandas DataFrame is required.")

    if features is None:
        features = [feature for feature in df.columns if pd.api.types.is_numeric_dtype(df[feature])]
    else:
        features = [feature for feature in features if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])]

    if len(features) < 2:
        raise ValueError("At least two numeric features are required for a projection plot.")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.45, s=20, color="steelblue")
    ax.set_title("Clustering Tendency Preview (PCA Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, X_pca


def plot_selected_feature_distributions(df: pd.DataFrame, features=None, figsize: tuple = (16, 8), show: bool = True):
    """
    Show the selected numeric features as grouped boxplots by machine type.

    This is a strong presentation visual because it highlights how operating signals differ
    across the categorical groups in the dataset.
    """
    if df is None:
        raise ValueError("A pandas DataFrame is required.")

    if features is None:
        features = list(df.columns)
    else:
        features = [feature for feature in features if feature in df.columns]

    numeric_features = [feature for feature in features if pd.api.types.is_numeric_dtype(df[feature])]
    if not numeric_features:
        raise ValueError("No numeric features were provided for distribution plotting.")

    if "Type" in df.columns and "Type" in features:
        n_cols = 2
        n_rows = max(1, (len(numeric_features) + 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
        fig.suptitle("Selected Feature Distributions by Type", fontsize=14, fontweight="bold")

        if n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        for ax, feature in zip(axes, numeric_features):
            sns.boxplot(data=df, x="Type", y=feature, ax=ax, color="lightsteelblue")
            ax.set_title(feature)
            ax.set_xlabel("Type")
            ax.set_ylabel("")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        for ax in axes[len(numeric_features):]:
            ax.remove()
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        sns.boxplot(data=df[numeric_features], orient="h", ax=axes)
        axes.set_title("Selected Feature Distributions")
        axes.set_xlabel("")
        axes.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()

    return fig
