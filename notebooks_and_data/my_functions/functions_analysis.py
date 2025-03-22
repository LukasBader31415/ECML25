import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from collections import Counter
from tqdm import tqdm

from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples, 
    calinski_harabasz_score, davies_bouldin_score, 
    jaccard_score
)

import hdbscan
import umap
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import MultiPolygon, Polygon
import re
import textwrap
from sklearn.ensemble import RandomForestClassifier


class DataPreparation:
    @staticmethod
    def prepare_us_geometries_with_original_names(state_path, county_path):
        """
        Lädt und bereitet US-Bundesstaat- und County-Geometrien für Karten und Analysen auf.
        Schließt Hawaii, Außengebiete und abgelegene Teile Alaskas aus.
    
        Rückgabe:
        - Ursprüngliche Bundesstaaten und Countys
        - Gefilterte Bundesstaaten & Countys für Kontinental-USA und Alaska
        """
        # Hilfsfunktion: Entfernt abgelegene Teile Alaskas
        def filter_alaska_polygons(gdf):
            alaska_idx = gdf[gdf['NAME'] == 'Alaska'].index
            if alaska_idx.empty:
                return gdf
    
            alaska_geometry = gdf.loc[alaska_idx[0], 'geometry']
            updated_geometry = None
    
            if isinstance(alaska_geometry, MultiPolygon):
                filtered_polys = [
                    poly for poly in alaska_geometry.geoms 
                    if poly.exterior.coords[0][0] <= 150
                ]
                if filtered_polys:
                    updated_geometry = MultiPolygon(filtered_polys)
            elif isinstance(alaska_geometry, Polygon):
                if alaska_geometry.exterior.coords[0][0] <= 150:
                    updated_geometry = alaska_geometry
    
            if updated_geometry is not None:
                gdf.loc[alaska_idx, 'geometry'] = updated_geometry
            else:
                gdf = gdf.drop(index=alaska_idx)
    
            return gdf
    
        # Laden der Pickle-Dateien
        state_shape = pd.read_pickle(state_path)
        county_shape = pd.read_pickle(county_path)
    
        # Entferne abgelegene Alaska-Geometrien
        state_shape = filter_alaska_polygons(state_shape)
    
        # Rename GEOID zu FIPS, falls notwendig
        if 'GEOID' in county_shape.columns:
            county_shape = county_shape.rename(columns={'GEOID': 'FIPS'})
    
        # Staaten, die nicht zum Festland gehören
        excluded_states = [
            'Hawaii',
            'Puerto Rico',
            'Commonwealth of the Northern Mariana Islands',
            'American Samoa',
            'United States Virgin Islands',
            'Guam'
        ]
    
        # STATEFP-Codes für Festland-USA und Alaska
        filtered_statefp_north_america = state_shape.loc[
            ~state_shape['NAME'].isin(excluded_states), 'STATEFP'
        ]
        filtered_statefp_alaska = state_shape.loc[
            state_shape['NAME'] == 'Alaska', 'STATEFP'
        ]
    
        # Gefilterte State-Geometrien
        filtered_state_shape_north_america = state_shape[
            state_shape['STATEFP'].isin(filtered_statefp_north_america)
        ]
        filtered_state_shape_alaska = state_shape[
            state_shape['STATEFP'].isin(filtered_statefp_alaska)
        ]
    
        # Gefilterte County-Geometrien (inkl. Entfernen von St. Paul Island)
        filtered_county_shape_north_america = county_shape[
            county_shape['STATEFP'].isin(filtered_statefp_north_america) &
            (county_shape['FIPS'] != '02016')
        ]
        filtered_county_shape_alaska = county_shape[
            county_shape['STATEFP'].isin(filtered_statefp_alaska) &
            (county_shape['FIPS'] != '02016')
        ]
    
        return (
            state_shape,
            county_shape,
            filtered_state_shape_north_america,
            filtered_state_shape_alaska,
            filtered_county_shape_north_america,
            filtered_county_shape_alaska
        )

    @staticmethod
    def assign_industry_regions(filtered_state_shape_north_america):
        """
        Assigns U.S. states to broader industry regions based on GEOID,
        and identifies any states assigned to more than one region.
    
        Parameters:
        - filtered_state_shape_north_america (GeoDataFrame): Cleaned GeoDataFrame of continental U.S. states.
    
        Returns:
        - Tuple of:
            - GeoDataFrame with industry region assignment
            - List of states assigned to multiple regions
        """
    
        # Define region classification data as CSV string
        data = """
    State ID,GEOID,State,Region
    IL,17,Illinois,North-Eastern
    KS,20,Kansas,Southern
    IN,18,Indiana,North-Eastern
    MI,26,Michigan,North-Eastern
    OH,39,Ohio,North-Eastern
    WI,55,Wisconsin,North-Eastern
    CT,09,Connecticut,New England
    ME,23,Maine,The New England
    MA,25,Massachusetts,New England
    NH,33,New Hampshire,New England
    RI,44,Rhode Island,New England
    VT,50,Vermont,New England
    DE,10,Delaware,New York and Mid-Atlantic
    DC,11,District of Columbia,New York and Mid-Atlantic
    MD,24,Maryland,New York and Mid-Atlantic
    NJ,34,New Jersey,New York and Mid-Atlantic
    NY,36,New York,New York and Mid-Atlantic
    PA,42,Pennsylvania,New York and Mid-Atlantic
    CA,06,California,Pacific Coastal
    OR,41,Oregon,Pacific Coastal
    WA,53,Washington,Pacific Coastal
    AL,01,Alabama,Southern
    AR,05,Arkansas,Southern
    FL,12,Florida,Southern
    GA,13,Georgia,Southern
    LA,22,Louisiana,Southern
    MS,28,Mississippi,Southern
    OK,40,Oklahoma,Southern
    TN,47,Tennessee,Southern
    TX,48,Texas,Southern
    AZ,04,Arizona,Western
    CO,08,Colorado,Western
    NV,32,Nevada,Western
    NM,35,New Mexico,Western
    UT,49,Utah,Western
    WY,56,Wyoming,Western
    PR,72,Puerto Rico,Other
    AS,60,American Samoa,Other
    AK,02,Alaska,Other
    HI,15,Hawaii,Other
    GU,66,Guam,Other
    VI,78,U.S. Virgin Islands,Other
    MP,69,Northern Mariana Islands,Other
    """
    
        # Read the region data into a DataFrame
        region_df = pd.read_csv(StringIO(data), dtype={'GEOID': str})
    
        # Merge with state geometries to assign regions
        industry_regions = filtered_state_shape_north_america.merge(region_df, on="GEOID", how="left")
    
        # Fill missing region entries with 'Other'
        industry_regions['Region'] = industry_regions['Region'].fillna('Other')
    
        # Find states assigned to more than one region
        multi_region_states = (
            industry_regions.groupby("State")["Region"]
            .nunique()
            .loc[lambda x: x > 1]
            .index
            .tolist()
        )
    
        return industry_regions, multi_region_states

    @staticmethod
    def find_missing_us_territories(county_master, industry_regions):
        """
        Identifies U.S. territories or states from the master dataset that are
        not included in the industry_regions DataFrame (e.g., excluded areas like Puerto Rico or Guam).
    
        Parameters:
        - county_master (DataFrame): Master DataFrame containing at least 'statefp' and 'state_name'.
        - industry_regions (GeoDataFrame or DataFrame): DataFrame containing at least 'STATEFP'.
    
        Returns:
        - Tuple of:
            - DataFrame with unique missing entries (statefp and state_name)
            - List of missing statefp values
        """
    
        # Merge to find which statefps from county_master are not in industry_regions
        merged = pd.merge(
            county_master[['statefp', 'state_name']],
            industry_regions[['STATEFP']],
            left_on='statefp',
            right_on='STATEFP',
            how='left',
            indicator=True
        )
    
        # Filter for entries only in county_master
        missing_entries = merged[merged['_merge'] == 'left_only']
    
        # Drop duplicates and keep only relevant columns
        unique_missing = missing_entries[['statefp', 'state_name']].drop_duplicates()
    
        # Extract list of missing statefp values
        missing_statefps = unique_missing['statefp'].to_list()
    
        return unique_missing, missing_statefps

    @staticmethod
    def load_and_filter_feature_dataframe(df_path, excluded_statefps):
        """
        Filters out FIPS entries from U.S. territories not included in the analysis.
        Compares the number of unique FIPS codes before and after filtering.
    
        Parameters:
        - df_path (str): Path to the pickled DataFrame containing a 'FIPS' column.
        - excluded_statefps (list): List of state FIPS codes (as strings) to exclude.
    
        Returns:
        - DataFrame: Filtered DataFrame with excluded territories removed.
        - Tuple of (original FIPS count, filtered FIPS count)
        """
    
        # Load original DataFrame
        df = pd.read_pickle(df_path)
    
        # Count original number of unique FIPS codes
        original_count = df['FIPS'].nunique()
    
        # Filter out rows where the first two digits of the FIPS code match excluded territories
        df_filtered = df[~df['FIPS'].str[:2].isin(excluded_statefps)].reset_index(drop=True)
    
        # Count number of unique FIPS codes after filtering
        filtered_count = df_filtered['FIPS'].nunique()
        print(f"Unique FIPS before: {original_count}")
        print(f"Unique FIPS after: {filtered_count}")
        return df_filtered

    @staticmethod
    def apply_log10_scaling(df, id_columns=['FIPS'], fill_value=0.001):
        """
        Applies log10 scaling to all feature columns in the DataFrame, 
        handling zeros and NaN values to prevent math errors.
    
        Parameters:
        - df (DataFrame): The input DataFrame with both ID and feature columns.
        - id_columns (list of str): Columns to exclude from scaling (e.g., ['FIPS']).
        - fill_value (float): Small positive value to replace 0 and NaN before applying log10.
    
        Returns:
        - Tuple of:
            - df_scaled_log (DataFrame): Log10-scaled DataFrame including ID columns.
            - df_original_features (DataFrame): The original feature columns (excluding ID columns).
        """
    
        # Save ID columns separately
        df_ids = df[id_columns].copy()
    
        # Drop ID columns to isolate features
        df_features = df.drop(columns=id_columns)
    
        # Replace 0 and NaN with a small positive value to avoid log10 errors
        df_features = df_features.replace(0, fill_value).fillna(fill_value)
    
        # Apply log10 transformation
        df_scaled_features_log = np.log10(df_features)
    
        # Combine transformed features with ID columns
        df_scaled_log = pd.concat([df_ids, df_scaled_features_log], axis=1)
    
        # Also return original (non-log) feature values if needed
        df_original_features = df_features.copy()
    
        return df_scaled_log, df_scaled_features_log, df_original_features


    @staticmethod
    def plot_feature_distributions_before_after_log(
        df_original, df_scaled_log, id_column='FIPS', marker_size=10, figsize=(9, 3)
    ):
        """
        Compares the distribution of feature values before and after log10 scaling using scatter plots.
    
        Parameters:
        - df_original (DataFrame): Original DataFrame including ID column and raw feature values.
        - df_scaled_log (DataFrame): Log10-scaled DataFrame (same structure as df_original).
        - id_column (str): Name of the identifier column to exclude from plotting (e.g., 'FIPS').
        - marker_size (int): Size of scatter plot markers.
        - figsize (tuple): Size of the figure.
    
        Returns:
        - None (plots the comparison figure)
        """
    
        # Get feature columns (exclude ID column)
        columns_original = df_original.drop(columns=[id_column]).columns
        df_features = df_original[columns_original]
    
        # Create feature IDs like F1, F2, F3, ...
        feature_id_dict = {feature: "F" + str(i + 1) for i, feature in enumerate(columns_original)}
        feature_ids = [feature_id_dict[col] for col in columns_original]
    
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)
    
        # Original feature scatter plot
        for i, col in enumerate(columns_original):
            axs[0].scatter(np.full(len(df_original[col]), i), df_original[col], alpha=0.5, s=marker_size)
        axs[0].set_title('Original Features')
        axs[0].set_xticks(range(len(columns_original)))
        axs[0].set_xticklabels(feature_ids, rotation=90)
        axs[0].set_ylabel('Value')
        for i, col in enumerate(columns_original):
            axs[0].set_ylim(df_original[col].min() - 2000, df_original[col].max() + 2000)
    
        # Scaled feature scatter plot
        for i, col in enumerate(columns_original):
            axs[1].scatter(np.full(len(df_scaled_log[col]), i), df_scaled_log[col], alpha=0.5, s=marker_size)
        axs[1].set_title('Log10 Scaled Features')
        axs[1].set_xticks(range(len(columns_original)))
        axs[1].set_xticklabels(feature_ids, rotation=90)
        for i, col in enumerate(columns_original):
            axs[1].set_ylim(df_scaled_log[col].min() - 1., df_scaled_log[col].max() + 2)
    
        plt.tight_layout()
        plt.show()




    @staticmethod
    def plot_histograms(df_original, df_scaled_log, id_column='FIPS', bins=30):
        """
        Plots side-by-side histograms for each feature in the dataset,
        comparing the original and log10-scaled distributions.
    
        Parameters:
        - df_original (DataFrame): DataFrame with original (unscaled) feature values.
        - df_scaled_log (DataFrame): DataFrame with log10-scaled feature values.
        - id_column (str): Column name to exclude from features (default: 'FIPS').
        - bins (int): Number of bins to use in histograms (default: 30).
    
        Returns:
        - None (shows plots for each feature)
        """
    
        # Get feature columns by excluding the identifier
        feature_columns = [col for col in df_original.columns if col != id_column]
    
        # Plot each feature's distribution before and after log-scaling
        for col in feature_columns:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original distribution
            axs[0].hist(df_original[col], bins=bins, alpha=0.7, edgecolor='black')
            axs[0].set_title(f'{col} - Original')
            axs[0].set_xlabel('Value')
            axs[0].set_ylabel('Frequency')
            
            # Log10-scaled distribution
            axs[1].hist(df_scaled_log[col], bins=bins, alpha=0.7, edgecolor='black')
            axs[1].set_title(f'{col} - Log10 Scaled')
            axs[1].set_xlabel('Value')
            axs[1].set_ylabel('Frequency')
            
            # Super title for the combined plot
            fig.suptitle(f'Distributions for {col}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    @staticmethod
    def create_feature_summary_table(df_original, id_column='FIPS', descriptions=None, fallback_count=3233):
        """
        Creates a structured summary table for all features, including:
        - Feature metadata (name, ID, description)
        - Data types
        - Statistical properties (min, max, mean, std)
        - Coverage (number of counties with values > 0 and not NaN)
    
        Parameters:
        - df_original (DataFrame): DataFrame with ID and feature columns (unscaled and unfilled).
        - id_column (str): Name of the identifier column to exclude from analysis.
        - descriptions (list of str or None): Optional list of descriptions for features.
        - fallback_count (int): Fallback value if coverage cannot be calculated.
    
        Returns:
        - summary_df (DataFrame): Unstyled summary table with statistics and metadata.
        - styled_summary (Styler): Styled summary table for display in notebooks.
        """
    
        # Isolate features (drop ID column)
        df_features = df_original.drop(columns=[id_column])
        
        # Generate feature IDs like F1, F2, ...
        feature_ids = [f"F{i+1}" for i in range(df_features.shape[1])]
    
        # Use empty descriptions if none provided
        if descriptions is None:
            descriptions = [''] * df_features.shape[1]
    
        # Build summary DataFrame
        summary_df = pd.DataFrame({
            'Feature': df_features.columns,
            'ID': feature_ids,
            'Description': descriptions,
            'Data Type': df_features.dtypes.replace('float64', 'Numeric')
                                            .replace('int64', 'Numeric')
                                            .replace('object', 'Binary'),
            'min': df_features.min(),
            'max': df_features.max(),
            'mean': df_features.mean(),
            'std': df_features.std(),
            'Number of counties': df_features.apply(lambda x: ((x > 0) & x.notna()).sum())
        }).reset_index(drop=True)
    
        # Replace NaNs in stats with 'N/A'
        summary_df[['min', 'max', 'mean', 'std']] = summary_df[['min', 'max', 'mean', 'std']].fillna('N/A')
    
        # Convert min/max to int if possible
        summary_df['min'] = summary_df['min'].apply(lambda x: int(x) if isinstance(x, (int, float)) else x)
        summary_df['max'] = summary_df['max'].apply(lambda x: int(x) if isinstance(x, (int, float)) else x)
    
        # Format mean and std to 1 decimal place
        summary_df['mean'] = summary_df['mean'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
        summary_df['std'] = summary_df['std'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    
        # Fill missing count values and ensure integer type
        summary_df['Number of counties'] = summary_df['Number of counties'].fillna(fallback_count).astype(int)
    
        # Style the summary table for display
        styled_summary = summary_df.style.set_properties(**{'text-align': 'left'}) \
                                         .set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
    
        return summary_df, styled_summary

class DimensionalityReduction:
    @staticmethod
    def plot_tsne_2d(df_features, perplexity=40, learning_rate=200, n_iter=500, random_state=42, 
                     point_size=5, color='blue', title='DR t-SNE', labels=None, figsize=(8, 6)):
        """
        Runs t-SNE on high-dimensional data and visualizes it in 2D space.
    
        Parameters:
        - df_features (DataFrame): DataFrame of scaled feature values (no IDs).
        - perplexity (int): t-SNE perplexity (default: 40).
        - learning_rate (int): t-SNE learning rate (default: 200).
        - n_iter (int): Number of optimization iterations (default: 500).
        - random_state (int): Random seed for reproducibility (default: 42).
        - point_size (int): Size of scatter plot points (default: 5).
        - color (str or list): Color of points, or array of colors per point.
        - title (str): Plot title (default: 'DR t-SNE').
        - labels (array-like, optional): Optional labels for legend (e.g., clusters).
        - figsize (tuple): Size of the figure (default: (8, 6)).
    
        Returns:
        - X_tsne (ndarray): 2D array of t-SNE transformed coordinates.
        """
    
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                    n_iter=n_iter, random_state=random_state)
        X_tsne = tsne.fit_transform(df_features)
    
        # Create scatter plot
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # If labels are provided, color by label group
            for label in set(labels):
                idx = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=point_size, label=str(label), alpha=0.6)
            plt.legend(title="Label")
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=point_size, color=color, alpha=0.6)
    
        plt.title(title)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.show()
    
        return X_tsne

    @staticmethod
    def plot_umap_2d(df_features, n_components=2, random_state=42, point_size=5,
                     color='blue', title='DR UMAP', labels=None, figsize=(8, 6)):
        """
        Reduces high-dimensional data to 2D using UMAP and visualizes the result.
    
        Parameters:
        - df_features (DataFrame): Scaled feature DataFrame (no IDs).
        - n_components (int): Number of UMAP output dimensions (default: 2).
        - random_state (int): Random seed for reproducibility (default: 42).
        - point_size (int): Size of scatter plot markers (default: 5).
        - color (str or list): Color or list of colors for plotting.
        - title (str): Title for the plot (default: 'DR UMAP').
        - labels (array-like, optional): Optional grouping labels for color grouping.
        - figsize (tuple): Size of the figure (default: (8, 6)).
    
        Returns:
        - embedding (ndarray): UMAP-reduced 2D coordinates.
        """
    
        # Fix randomness
        np.random.seed(random_state)
    
        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(df_features)
    
        # Create the plot
        plt.figure(figsize=figsize)
    
        if labels is not None:
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels,
                            palette='tab10', s=point_size, alpha=0.7, legend='full')
            plt.legend(title="Label")
        else:
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], s=point_size,
                            color=color, alpha=0.7)
    
        plt.title(title)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.show()
    
        return embedding

    @staticmethod
    def plot_pca_2d_3d(df_scaled, id_column='FIPS', labels=None, color='blue', point_size=5):
        """
        Applies PCA to reduce high-dimensional data to 2D and 3D and visualizes both projections.
    
        Parameters:
        - df_scaled (DataFrame): Scaled DataFrame including an ID column (e.g., 'FIPS').
        - id_column (str): Name of ID column to exclude from PCA (default: 'FIPS').
        - labels (array-like, optional): Optional labels for grouping (used in legends).
        - color (str or list): Color or list of colors for plotting points.
        - point_size (int): Size of the scatter points.
    
        Returns:
        - Tuple of (X_pca_2d, X_pca_3d): The PCA-transformed arrays.
        """
    
        # Drop ID column
        X = df_scaled.drop(columns=[id_column])
    
        # PCA transformation
        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)
        X_pca_2d = pca_2d.fit_transform(X)
        X_pca_3d = pca_3d.fit_transform(X)
    
        # Create subplot layout
        fig = plt.figure(figsize=(14, 6))
    
        # 2D PCA plot
        ax1 = fig.add_subplot(121)
        if labels is not None:
            for label in set(labels):
                idx = [i for i, l in enumerate(labels) if l == label]
                ax1.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1], s=point_size, label=str(label), alpha=0.6)
            ax1.legend(title="Label")
        else:
            ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=point_size, color=color, alpha=0.6)
    
        ax1.set_title('PCA Visualization (2D)')
        ax1.set_xlabel('PCA 1')
        ax1.set_ylabel('PCA 2')
    
        # 3D PCA plot
        ax2 = fig.add_subplot(122, projection='3d')
        if labels is not None:
            for label in set(labels):
                idx = [i for i, l in enumerate(labels) if l == label]
                ax2.scatter(X_pca_3d[idx, 0], X_pca_3d[idx, 1], X_pca_3d[idx, 2], 
                            s=point_size, label=str(label), alpha=0.6)
            ax2.legend(title="Label")
        else:
            ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                        s=point_size, color=color, alpha=0.6)
    
        ax2.set_title('PCA Visualization (3D)')
        ax2.set_xlabel('PCA 1')
        ax2.set_ylabel('PCA 2')
        ax2.set_zlabel('PCA 3')
    
        plt.tight_layout()
        plt.show()
    
        return X_pca_2d, X_pca_3d

class Clustering:
    @staticmethod
    def run_hdbscan_gridsearch(df_scaled,param_grid,id_column='FIPS',true_labels=None,save_path=None):
        """
        Performs a grid search over HDBSCAN hyperparameters and evaluates internal clustering metrics.
    
        Parameters:
        - df_scaled (DataFrame): Scaled feature DataFrame including ID column.
        - param_grid (dict): Dictionary with 'min_cluster_size' and 'min_samples' ranges.
        - id_column (str): Name of the ID column to exclude from features (default: 'FIPS').
        - true_labels (array-like, optional): Ground truth labels for optional Jaccard scoring.
        - save_path (str, optional): If given, path to save the results as pickle.
    
        Returns:
        - results_df (DataFrame): DataFrame with metrics for each parameter combination.
        """
    
        data_scaled = df_scaled.drop(columns=[id_column])
        hdbscan_results = []
    
        for min_cluster_size in tqdm(param_grid['min_cluster_size'], desc="min_cluster_size"):
            for min_samples in tqdm(param_grid['min_samples'], desc="min_samples", leave=False):
    
                model = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples)
                )
                labels = model.fit_predict(data_scaled)
                unique_clusters = len(set(labels) - {-1})
    
                if unique_clusters > 1:
                    silhouette = silhouette_score(data_scaled, labels)
                    calinski = metrics.calinski_harabasz_score(data_scaled, labels)
                    davies = metrics.davies_bouldin_score(data_scaled, labels)
                    sil_samples = silhouette_samples(data_scaled, labels)
                    neg_sil_count = np.sum(sil_samples < 0)
                else:
                    silhouette = -1
                    calinski = -1
                    davies = -1
                    sil_samples = np.array([])
                    neg_sil_count = 0
    
                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels)
                avg_persistence = np.mean(model.cluster_persistence_) if unique_clusters > 0 else 0
                avg_cluster_prob = np.mean(model.probabilities_) if hasattr(model, 'probabilities_') else np.nan
    
                if unique_clusters > 1 and true_labels is not None:
                    try:
                        jaccard = jaccard_score(true_labels, labels, average='macro')
                    except:
                        jaccard = -1
                else:
                    jaccard = -1
    
                hdbscan_results.append({
                    'min_cluster_size': int(min_cluster_size),
                    'min_samples': int(min_samples),
                    'unique_clusters': unique_clusters,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'negative_silhouette_count': neg_sil_count,
                    'noise_ratio': noise_ratio,
                    'hdbscan_persistence': avg_persistence,
                    'avg_cluster_probability': avg_cluster_prob,
                    'jaccard': jaccard
                })
    
        results_df = pd.DataFrame(hdbscan_results)
    
        if save_path:
            results_df.to_pickle(save_path)
    
        return results_df


    @staticmethod
    def select_best_hdbscan_result(
        result_path,
        noise_threshold=0.1,
        min_clusters=2,
        min_persistence=0.5
    ):
        """
        Selects the best HDBSCAN clustering result from a stored results file based on multiple criteria.
    
        Parameters:
        - result_path (str): Path to the HDBSCAN results pickle file.
        - noise_threshold (float): Maximum allowed noise ratio (default: 0.1).
        - min_clusters (int): Minimum number of clusters (default: 2).
        - min_persistence (float): Minimum average persistence score (default: 0.5).
    
        Returns:
        - best_result (dict): Dictionary with parameters and metrics of the selected best configuration.
        """
    
        # Load result DataFrame from disk
        results_df = pd.read_pickle(result_path)
        results_list = results_df.to_dict('records')
    
        # Filter based on evaluation criteria
        filtered = [
            r for r in results_list
            if r['unique_clusters'] >= min_clusters
            and r['noise_ratio'] < noise_threshold
            and r['hdbscan_persistence'] >= min_persistence
        ]
    
        # If no results match, fall back to all
        if not filtered:
            filtered = results_list
    
        # Select the best configuration using a weighted tuple of metrics
        best_result = max(
            filtered,
            key=lambda x: (
                x['silhouette'],                  # Prefer high silhouette
                -x['negative_silhouette_count'],  # Prefer fewer bad samples
                x['hdbscan_persistence'],         # Prefer stable clusters
                -x['noise_ratio']                 # Prefer low noise
            )
        )
    
        return best_result

    @staticmethod
    def run_hdbscan_gridsearch_tsne(
        tsne_embedding,
        param_grid,
        save_path=None
    ):
        """
        Performs a grid search for HDBSCAN on 2D t-SNE-reduced data.
        Evaluates clustering quality using internal metrics.
    
        Parameters:
        - tsne_embedding (ndarray or DataFrame): 2D t-SNE reduced data (shape: [n_samples, 2]).
        - param_grid (dict): Dict with 'min_cluster_size' and 'min_samples' ranges.
        - save_path (str, optional): If provided, path to save results as pickle.
    
        Returns:
        - results_df (DataFrame): DataFrame of clustering evaluation results.
        """
    
        results = []
    
        for min_cluster_size in tqdm(param_grid['min_cluster_size'], desc="min_cluster_size"):
            for min_samples in tqdm(param_grid['min_samples'], desc="min_samples", leave=False):
                model = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples)
                )
                labels = model.fit_predict(tsne_embedding)
    
                unique_clusters = len(set(labels) - {-1})
    
                if unique_clusters > 1:
                    silhouette = silhouette_score(tsne_embedding, labels)
                    calinski = metrics.calinski_harabasz_score(tsne_embedding, labels)
                    davies = metrics.davies_bouldin_score(tsne_embedding, labels)
                    sil_samples = silhouette_samples(tsne_embedding, labels)
                    neg_sil_count = np.sum(sil_samples < 0)
                else:
                    silhouette = -1
                    calinski = -1
                    davies = -1
                    sil_samples = np.array([])
                    neg_sil_count = 0
    
                noise_ratio = np.mean(labels == -1)
                avg_persistence = np.mean(model.cluster_persistence_) if unique_clusters > 0 else 0
                avg_cluster_prob = np.mean(model.probabilities_) if hasattr(model, 'probabilities_') else np.nan
    
                results.append({
                    'min_cluster_size': int(min_cluster_size),
                    'min_samples': int(min_samples),
                    'unique_clusters': unique_clusters,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'negative_silhouette_count': neg_sil_count,
                    'noise_ratio': noise_ratio,
                    'hdbscan_persistence': avg_persistence,
                    'avg_cluster_probability': avg_cluster_prob
                })
    
        results_df = pd.DataFrame(results)
    
        if save_path:
            results_df.to_pickle(save_path)
    
        return results_df

    @staticmethod
    def select_best_hdbscan_reduced_result(
        result_path,
        noise_threshold=0.1,
        min_clusters=2,
        min_persistence=0.5
    ):
        """
        Selects the best HDBSCAN clustering result after dimensionality reduction
        (e.g., t-SNE or UMAP), based on multiple internal metrics.
    
        Parameters:
        - result_path (str): Path to the pickle file containing HDBSCAN evaluation results.
        - noise_threshold (float): Maximum allowed noise ratio (default: 0.1).
        - min_clusters (int): Minimum number of valid clusters required (default: 2).
        - min_persistence (float): Minimum average cluster persistence (default: 0.5).
    
        Returns:
        - best_result (dict): Dictionary with the best result's metrics and parameters.
        """
    
        # Load result DataFrame from disk
        results_df = pd.read_pickle(result_path)
        results_list = results_df.to_dict('records')
    
        # Filter based on evaluation thresholds
        filtered = [
            r for r in results_list
            if r['unique_clusters'] >= min_clusters
            and r['noise_ratio'] < noise_threshold
            and r['hdbscan_persistence'] >= min_persistence
        ]
    
        # Fall back to all results if none pass the filter
        if not filtered:
            filtered = results_list
    
        # Select best based on multiple internal metrics
        best_result = max(
            filtered,
            key=lambda x: (
                x['silhouette'],                  # High silhouette
                -x['negative_silhouette_count'],  # Fewer negative silhouette samples
                x['hdbscan_persistence'],         # Higher stability
                -x['noise_ratio']                 # Lower noise
            )
        )
    
        return best_result



    @staticmethod
    def run_dbscan_gridsearch_tsne(
        tsne_data,
        param_grid,
        save_path=None
    ):
        """
        Performs a grid search over DBSCAN parameters on 2D t-SNE-reduced data
        and evaluates clustering quality using internal metrics.
    
        Parameters:
        - tsne_data (array-like): 2D t-SNE-reduced data (shape: [n_samples, 2]).
        - param_grid (dict): Dict with 'eps' (array-like) and 'min_samples' (iterable).
        - save_path (str, optional): If provided, saves the results DataFrame to this path.
    
        Returns:
        - dbscan_results_df (DataFrame): DataFrame containing clustering evaluation metrics.
        """
    
        results = []
    
        for eps in tqdm(param_grid['eps'], desc="eps"):
            for min_samples in tqdm(param_grid['min_samples'], desc="min_samples", leave=False):
                model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                labels = model.fit_predict(tsne_data)
    
                unique_clusters = len(set(labels) - {-1})
    
                if unique_clusters > 1:
                    silhouette = silhouette_score(tsne_data, labels)
                    silhouette_avg = np.mean(silhouette_samples(tsne_data, labels))
                    calinski = metrics.calinski_harabasz_score(tsne_data, labels)
                    davies = metrics.davies_bouldin_score(tsne_data, labels)
                    sil_samples = silhouette_samples(tsne_data, labels)
                    neg_sil_count = np.sum(sil_samples < 0)
                else:
                    silhouette = -1
                    silhouette_avg = -1
                    calinski = -1
                    davies = -1
                    sil_samples = np.array([])
                    neg_sil_count = 0
    
                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels)
    
                results.append({
                    'eps': float(eps),
                    'min_samples': int(min_samples),
                    'unique_clusters': unique_clusters,
                    'silhouette': silhouette,
                    'silhouette_avg': silhouette_avg,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'negative_silhouette_count': neg_sil_count,
                    'noise_ratio': noise_ratio
                })
    
        dbscan_results_df = pd.DataFrame(results)
    
        if save_path:
            dbscan_results_df.to_pickle(save_path)
    
        return dbscan_results_df

    @staticmethod
    def select_best_dbscan_result(
        result_path,
        min_clusters=2,
        noise_threshold=0.2,
        strict_noise_preference=0.05
    ):
        """
        Selects the best DBSCAN clustering result based on internal metrics, after dimensionality reduction.
    
        Parameters:
        - result_path (str): Path to the saved DBSCAN result pickle file.
        - min_clusters (int): Minimum number of clusters required (default: 2).
        - noise_threshold (float): Maximum noise ratio for filtering candidates (default: 0.2).
        - strict_noise_preference (float): Optional tighter noise preference during ranking (default: 0.05).
    
        Returns:
        - best_result (dict): Dictionary containing metrics and parameters of the best DBSCAN result.
        """
    
        # Load results
        df = pd.read_pickle(result_path)
        results_list = df.to_dict('records')
    
        # Filter by thresholds
        filtered = [
            r for r in results_list
            if r['unique_clusters'] >= min_clusters and r['noise_ratio'] < noise_threshold
        ]
    
        # Fallback to all if none match
        if not filtered:
            filtered = results_list
    
        # Select best based on custom scoring
        best_result = max(
            filtered,
            key=lambda x: (
                -x['negative_silhouette_count'],  # Fewer bad silhouette values
                -x['noise_ratio'],                # Lower noise
                x['silhouette_avg']               # Better average separation
            )
        )
    
        return best_result

    @staticmethod
    def apply_and_color_clusters(
        df_scaled,
        X_tsne,
        hdbscan_full_space_best_result,
        hdbscan_reduced_space_best_result,
        dbscan_best_params={'eps': 4, 'min_samples': 35}
    ):
        """
        Applies HDBSCAN and DBSCAN clustering on full and t-SNE-reduced feature spaces,
        and assigns cluster labels and color codes to each data point.
    
        Parameters:
        - df_scaled (DataFrame): Scaled feature data including FIPS column.
        - X_tsne (ndarray): 2D t-SNE representation of the data.
        - hdbscan_full_space_best_result (dict): Best parameters from HDBSCAN on full space.
        - hdbscan_reduced_space_best_result (dict): Best parameters from HDBSCAN on t-SNE space.
        - dbscan_best_params (dict): Best parameters for DBSCAN (on t-SNE).
    
        Returns:
        - Tuple of DataFrames: (df_hdbscan_full, df_dbscan_tsne, df_hdbscan_tsne)
          Each with columns for cluster ID and assigned color.
        """
    
        def assign_colors(labels):
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            color_dict = {
                label: "#{:02x}{:02x}{:02x}".format(
                    int(cmap(i)[0] * 255),
                    int(cmap(i)[1] * 255),
                    int(cmap(i)[2] * 255)
                )
                for i, label in enumerate(unique_labels)
            }
            if -1 in color_dict:
                color_dict[-1] = "#0000FF"  # Blue for noise
            return color_dict
    
        # ---------------------------
        # HDBSCAN – Full Feature Space
        # ---------------------------
        hdb_full = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_full_space_best_result['min_cluster_size'],
            min_samples=hdbscan_full_space_best_result['min_samples']
        )
        labels_full = hdb_full.fit_predict(df_scaled.iloc[:, 1:])
        color_map_full = assign_colors(labels_full)
        df_hdbscan_full = df_scaled.iloc[:, 1:].copy()
        df_hdbscan_full['cluster_id'] = labels_full
        df_hdbscan_full['color'] = df_hdbscan_full['cluster_id'].map(color_map_full)
    
        # ---------------------------
        # DBSCAN – t-SNE Space
        # ---------------------------
        dbscan_model = DBSCAN(
            eps=dbscan_best_params['eps'],
            min_samples=dbscan_best_params['min_samples']
        )
        labels_dbscan = dbscan_model.fit_predict(X_tsne)
        color_map_dbscan = assign_colors(labels_dbscan)
        df_dbscan_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
        df_dbscan_tsne['cluster_id'] = labels_dbscan
        df_dbscan_tsne['color'] = df_dbscan_tsne['cluster_id'].map(color_map_dbscan)
    
        # ---------------------------
        # HDBSCAN – t-SNE Space
        # ---------------------------
        hdb_tsne = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_reduced_space_best_result['min_cluster_size'],
            min_samples=hdbscan_reduced_space_best_result['min_samples']
        )
        labels_hdb_tsne = hdb_tsne.fit_predict(X_tsne)
        color_map_hdb_tsne = assign_colors(labels_hdb_tsne)
        df_hdbscan_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
        df_hdbscan_tsne['cluster_id'] = labels_hdb_tsne
        df_hdbscan_tsne['color'] = df_hdbscan_tsne['cluster_id'].map(color_map_hdb_tsne)
    
        return df_hdbscan_full, df_dbscan_tsne, df_hdbscan_tsne

    @staticmethod
    def plot_cluster_comparison_tsne(
        X_tsne,
        df_dbscan_tsne,
        df_hdbscan_tsne,
        df_hdbscan_full,
        save_path='cluster_comparison_plot.png'
    ):
        """
        Visualizes and compares DBSCAN and HDBSCAN clustering results (full and t-SNE-reduced space)
        in a 2x2 subplot layout using t-SNE embedding.
    
        Parameters:
        - X_tsne (ndarray): 2D t-SNE embedding (shape: [n_samples, 2])
        - df_dbscan_tsne (DataFrame): Result of DBSCAN on t-SNE, with 'cluster_id' and 'color'
        - df_hdbscan_tsne (DataFrame): Result of HDBSCAN on t-SNE, with 'cluster_id' and 'color'
        - df_hdbscan_full (DataFrame): Result of HDBSCAN on full feature space, with 'cluster_id' and 'color'
        - save_path (str): Path to save the resulting plot (PNG by default)
    
        Returns:
        - None (saves and shows the plot)
        """
    
        def plot_clusters(ax, embedding, cluster_labels, counts, col_dict, title, method):
            for label in np.unique(cluster_labels):
                idx = (cluster_labels == label)
                label_name = f'Noise ({counts[label]})' if label == -1 else None
                ax.scatter(embedding[idx, 0], embedding[idx, 1],
                           color=col_dict[label], label=label_name, alpha=0.6, s=1)
    
            if -1 in counts:
                ax.legend(loc='lower left', fontsize=7)
    
            # Method label in upper-right corner
            ax.text(0.95, 0.90, method, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
            # Circle annotation for cluster IDs
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            radius = 0.029 * x_range
            y_offset = radius * -1.5
            for label in np.unique(cluster_labels):
                if label == -1:
                    continue
                idx = (cluster_labels == label)
                x_center = np.mean(embedding[idx, 0])
                y_center = np.mean(embedding[idx, 1])
                circle_center = (x_center, y_center + y_offset)
                circle = plt.Circle(circle_center, radius, color='white', ec='black', zorder=10)
                ax.add_patch(circle)
                ax.text(circle_center[0], circle_center[1], str(label),
                        fontsize=10, color='black',
                        horizontalalignment='center', verticalalignment='center', zorder=11)
    
        # Prepare cluster info
        labels_dbscan = df_dbscan_tsne['cluster_id'].values
        labels_hdb_tsne = df_hdbscan_tsne['cluster_id'].values
        labels_hdb_full = df_hdbscan_full['cluster_id'].values
    
        colors_dbscan = df_dbscan_tsne.set_index('cluster_id')['color'].to_dict()
        colors_hdb_tsne = df_hdbscan_tsne.set_index('cluster_id')['color'].to_dict()
        colors_hdb_full = df_hdbscan_full.set_index('cluster_id')['color'].to_dict()
    
        counts_dbscan = dict(Counter(labels_dbscan))
        counts_hdb_tsne = dict(Counter(labels_hdb_tsne))
        counts_hdb_full = dict(Counter(labels_hdb_full))
    
        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
        # Plot 1: t-SNE only (no clustering)
        axes[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], color="Blue", alpha=0.6, s=1)
        axes[0, 0].text(0.95, 0.9, 't-SNE components', transform=axes[0, 0].transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
        # Plot 2: DBSCAN in t-SNE space
        plot_clusters(axes[0, 1], X_tsne, labels_dbscan, counts_dbscan, colors_dbscan,
                      'DBSCAN Clustering (t-SNE space)', 'DBSCAN (reduced space)')
    
        # Plot 3: HDBSCAN in full space, visualized in t-SNE
        plot_clusters(axes[1, 0], X_tsne, labels_hdb_full, counts_hdb_full, colors_hdb_full,
                      'HDBSCAN Clustering (all features)', 'HDBSCAN (full feature set)')
    
        # Plot 4: HDBSCAN in t-SNE space
        plot_clusters(axes[1, 1], X_tsne, labels_hdb_tsne, counts_hdb_tsne, colors_hdb_tsne,
                      'HDBSCAN Clustering (t-SNE space)', 'HDBSCAN (reduced space)')
    
        # Clean layout
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
    
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.savefig(save_path, dpi=600)
        plt.show()

    @staticmethod
    def evaluate_clusterings(
        labels_dict,
        data_dict,
        compare_pairs=True
    ):
        """
        Evaluates clustering results using internal metrics (unsupervised)
        and pairwise comparisons (external metrics: ARI, NMI).
    
        Parameters:
        - labels_dict (dict): Dict with clustering method names as keys and label arrays as values.
        - data_dict (dict): Dict with method names as keys and corresponding input data (for silhouette).
        - compare_pairs (bool): Whether to compute ARI/NMI pairwise comparisons.
    
        Returns:
        - df_metrics (DataFrame): Internal metrics per method (silhouette, cluster count, etc.)
        - pairwise_df (DataFrame or None): Pairwise ARI/NMI comparisons (if compare_pairs=True)
        """
    
        def compute_metrics(labels, data=None):
            labels = np.array(labels)
            result = {}
    
            # Cluster count (excluding noise)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            result['num_clusters'] = n_clusters
    
            # Cluster sizes (excluding noise)
            counts = Counter(labels)
            noise = counts.pop(-1, 0)
            sizes = sorted(counts.values(), reverse=True)
            result['cluster_sizes'] = sizes
    
            # Noise fraction
            result['noise_fraction'] = noise / len(labels)
    
            # Silhouette score
            if data is not None and n_clusters >= 2:
                try:
                    sil = metrics.silhouette_score(data, labels)
                except Exception:
                    sil = None
            else:
                sil = None
            result['silhouette'] = sil
    
            return result
    
        # Internal metrics
        records = []
        for method, labels in labels_dict.items():
            data = data_dict.get(method, None)
            results = compute_metrics(labels, data)
            results['method'] = method
            records.append(results)
    
        df_metrics = pd.DataFrame(records).set_index('method')
    
        # External comparisons (pairwise ARI & NMI)
        pairwise_df = None
        if compare_pairs:
            rows = []
            pairs = list(labels_dict.items())
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    name1, labels1 = pairs[i]
                    name2, labels2 = pairs[j]
                    ari = metrics.adjusted_rand_score(labels1, labels2)
                    nmi = metrics.normalized_mutual_info_score(labels1, labels2)
                    rows.append({
                        'Comparison': f'{name1} vs {name2}',
                        'ARI': ari,
                        'NMI': nmi
                    })
            pairwise_df = pd.DataFrame(rows).set_index('Comparison')
    
        return df_metrics, pairwise_df

class Vizualization:
    @staticmethod
    def plot_tsne_feature_maps(
        X_tsne,
        df_features,
        feature_id_prefix="F",
        save_path=None,
        cmap='viridis',
        point_size=5,
        alpha=0.5
    ):
        """
        Plots scatterplots of t-SNE embeddings colored by each feature value.
    
        Parameters:
        - X_tsne (ndarray): 2D t-SNE embedding (shape: [n_samples, 2])
        - df_features (DataFrame): DataFrame with feature values (no ID column)
        - feature_id_prefix (str): Prefix for feature IDs (default: 'F')
        - save_path (str or None): If given, saves plots to this folder
        - cmap (str): Matplotlib colormap for scatter
        - point_size (int): Size of scatter points
        - alpha (float): Transparency of points
    
        Returns:
        - None (displays or saves plots)
        """
    
        # Create feature-ID mapping
        feature_id_dict = {col: f"{feature_id_prefix}{i+1}" for i, col in enumerate(df_features.columns)}
    
        for col in df_features.columns:
            plt.figure(figsize=(8, 5))
            scatter = plt.scatter(
                X_tsne[:, 0], X_tsne[:, 1],
                c=df_features[col],
                cmap=cmap,
                s=point_size,
                alpha=alpha
            )
    
            # Colorbar and label
            cbar = plt.colorbar(scatter, label=col)
            label_text = feature_id_dict[col]
    
            # Text box bottom right
            plt.text(0.95, 0.05, label_text,
                     transform=plt.gca().transAxes,
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     fontsize=20,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f't-SNE colored by {col}')
            plt.tight_layout()
    
            if save_path:
                filename = f"{save_path}/tsne_feature_{label_text}.png"
                plt.savefig(filename, dpi=300)
            else:
                plt.show()
    
            plt.close()


    @staticmethod
    def plot_tsne_feature_grid(
        X_tsne,
        df_features,
        features_to_plot,
        feature_id_dict=None,
        save_path=None,
        cmap='viridis',
        point_size=2,
        alpha=0.5
    ):
        """
        Plots a 2x2 grid of t-SNE embeddings colored by selected feature values,
        using a shared colormap and annotated feature IDs.
    
        Parameters:
        - X_tsne (ndarray): 2D t-SNE array (shape: [n_samples, 2])
        - df_features (DataFrame): Feature DataFrame (scaled)
        - features_to_plot (list): List of 4 feature column names
        - feature_id_dict (dict): Optional dict mapping feature names to short labels (e.g., F1)
        - save_path (str or None): Path to save the figure (e.g., 'output.png'); if None, just show
        - cmap (str): Colormap for all scatterplots
        - point_size (int): Size of scatter points
        - alpha (float): Transparency
    
        Returns:
        - None
        """
    
        assert len(features_to_plot) == 4, "Exactly 4 features required for a 2x2 grid."
    
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
    
        # Shared color scale
        all_data = np.concatenate([df_features[col].values for col in features_to_plot])
        vmin, vmax = all_data.min(), all_data.max()
    
        for i, col in enumerate(features_to_plot):
            ax = axs[i]
            data = df_features[col]
            label_text = feature_id_dict[col] if feature_id_dict else col
    
            scatter = ax.scatter(
                X_tsne[:, 0], X_tsne[:, 1],
                c=data,
                cmap=cmap,
                alpha=alpha,
                s=point_size,
                vmin=vmin,
                vmax=vmax
            )
    
            ax.text(0.95, 0.05, label_text,
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=15,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
            ax.set_xticks([])
            ax.set_yticks([])
    
        # Shared colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(axs[0].collections[0], cax=cbar_ax, orientation='horizontal')
    
        # Layout adjustments
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
    
        if save_path:
            plt.savefig(save_path, dpi=600)
            plt.show()
    
        plt.close()

    @staticmethod
    def compute_cluster_rankings(
        df_scaled,
        cluster_df,
        cluster_col='cluster_id',
        color_col='color',
        exclude_columns=3
    ):
        """
        Computes mean values per cluster, ranks clusters by overall mean,
        and merges results back into the main DataFrame.
    
        Parameters:
        - df_scaled (DataFrame): Main DataFrame with scaled numeric features.
        - cluster_df (DataFrame): DataFrame containing 'cluster_id' and optional 'color'.
        - cluster_col (str): Name of the cluster ID column.
        - color_col (str): Name of the color column.
        - exclude_columns (int): Number of columns to exclude at the end (e.g., for meta-data like color).
    
        Returns:
        - df_scaled (DataFrame): Updated with 'cluster_id', 'color', 'rank', 'overall_mean' columns.
        - cluster_summary (DataFrame): Cluster-level summary with mean features, overall_mean, and rank.
        """
    
        # Merge cluster labels and color into df_scaled
        df_scaled = df_scaled.copy()
        df_scaled[cluster_col] = cluster_df[cluster_col].values
        if color_col in cluster_df.columns:
            df_scaled[color_col] = cluster_df[color_col].values
    
        # Select numeric feature columns (excluding ID and meta columns)
        numeric_columns = df_scaled.columns[1:-exclude_columns].tolist()
    
        # Cluster-level mean values
        cluster_summary = df_scaled.groupby(cluster_col)[numeric_columns].mean()
    
        # Overall mean for sorting
        cluster_summary['overall_mean'] = cluster_summary.mean(axis=1)
    
        # Rank clusters (descending)
        cluster_summary['rank'] = cluster_summary['overall_mean'].rank(
            method='dense', ascending=False).astype(int)
    
        cluster_summary = cluster_summary.reset_index()
    
        # Merge back rank and overall_mean to df_scaled
        df_scaled = df_scaled.merge(
            cluster_summary[[cluster_col, 'rank']], on=cluster_col, how='left'
        )
    
        # Compute row-wise overall mean per sample
        df_scaled['overall_mean'] = df_scaled[numeric_columns].mean(axis=1)
    
        return df_scaled, cluster_summary

    @staticmethod
    def plot_clustered_us_counties(
        filtered_county_shape_north_america,
        filtered_county_shape_alaska,
        filtered_state_shape_north_america,
        df_scaled_ranked,
        alaska_inset_position=[0.68, 0.255, 0.275, 0.275],
        cluster_col='cluster_id',
        color_col='color',
        legend_title='Cluster ID',
        clip_alaska=True
    ):
        """
        Plots US counties with cluster coloring, including an inset for Alaska.
        """
    
        # --- Merge cluster data with county shapes ---
        merged_counties = df_scaled_ranked[['FIPS', cluster_col, color_col, 'rank', 'overall_mean']]
    
        filtered_county_shape_north_america_merged = filtered_county_shape_north_america.merge(
            merged_counties, on='FIPS', how='left'
        )
        filtered_county_shape_alaska_merged = filtered_county_shape_alaska.merge(
            merged_counties, on='FIPS', how='left'
        )
    
        # Split Alaska and the rest of the US
        alaska_state = filtered_state_shape_north_america[filtered_state_shape_north_america['STATEFP'] == '02']
        alaska_counties = filtered_county_shape_north_america_merged[
            filtered_county_shape_north_america_merged['STATEFP'] == '02'
        ]
    
        remaining_states = filtered_state_shape_north_america[
            filtered_state_shape_north_america['STATEFP'] != '02'
        ]
        remaining_counties = filtered_county_shape_north_america_merged[
            filtered_county_shape_north_america_merged['STATEFP'] != '02'
        ]
    
        # --- Plot setup ---
        fig, ax = plt.subplots(figsize=(15, 7))
    
        # Plot continental US counties
        remaining_counties.plot(
            facecolor=remaining_counties[color_col],
            linewidth=0.4,
            ax=ax,
            edgecolor='grey',
            legend=False
        )
        # Plot state borders
        remaining_states.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
    
        # Create Alaska inset
        ak_ax = fig.add_axes(alaska_inset_position)
    
        if clip_alaska:
            # Clip Alaska shape to focus on main landmass
            polygon = Polygon([(-180, 50), (-180, 72), (-140, 72), (-140, 50)])
            alaska_state = alaska_state.clip(polygon)
            alaska_counties = alaska_counties.clip(polygon)
    
        # Plot Alaska counties and state borders
        alaska_counties.plot(
            facecolor=alaska_counties[color_col],
            linewidth=0.4,
            ax=ak_ax,
            edgecolor='grey',
            legend=False
        )
        alaska_state.plot(ax=ak_ax, color='none', edgecolor='black', linewidth=0.8)
    
        # Turn off axes for both maps
        ak_ax.axis('off')
        ax.axis('off')
    
        # --- Create legend ---
        unique_clusters = sorted(remaining_counties[cluster_col].dropna().unique())
        cluster_counts = remaining_counties[cluster_col].value_counts()
        legend_elements = []
    
        for cluster in unique_clusters:
            if cluster == -1:
                count = cluster_counts.get(-1, 0)
                legend_elements.append(Patch(facecolor='blue', label=f'Noise ({count})'))
            else:
                color = remaining_counties.loc[remaining_counties[cluster_col] == cluster, color_col].values[0]
                count = cluster_counts.get(cluster, 0)
                legend_elements.append(Patch(facecolor=color, label=f'{cluster} ({count})'))
    
        # Add legend with 3 columns
        ax.legend(
            handles=legend_elements,
            title=legend_title,
            loc='lower left',
            title_fontsize=10,
            fontsize=8,
            ncol=3
        )
    
        plt.tight_layout()
        plt.show()
        return filtered_county_shape_north_america_merged, filtered_county_shape_alaska_merged


    @staticmethod
    def plot_top_ranked_and_noise_counties(
        filtered_county_shape_north_america_merged,
        filtered_state_shape_north_america,
        top_rank_threshold=5,
        noise_cluster_id=-1,
        top_noise_n=20,
        alaska_inset_position=[0.67, 0.255, 0.275, 0.275]
    ):
        """
        Plots top-ranked counties and top noise-cluster counties in the US with fixed-position Alaska inset.
        """
    
        # --- Filter top-ranked and top noise-cluster counties ---
        filtered_counties = filtered_county_shape_north_america_merged[
            (filtered_county_shape_north_america_merged['rank'] <= top_rank_threshold) | 
            (filtered_county_shape_north_america_merged['cluster_id'] == noise_cluster_id)
        ]
    
        top_rank = filtered_counties[filtered_counties['rank'] <= top_rank_threshold]
    
        cluster_minus_one_top = (
            filtered_counties[filtered_counties['cluster_id'] == noise_cluster_id]
            .sort_values(by='overall_mean', ascending=False)
            .head(top_noise_n)
        )
    
        filtered_counties = pd.concat([top_rank, cluster_minus_one_top]).drop_duplicates()
    
        # Copy full dataset for outline plotting
        all_counties = filtered_county_shape_north_america_merged.copy()
    
        # --- Split Alaska and remaining US ---
        alaska_state = filtered_state_shape_north_america[filtered_state_shape_north_america['STATEFP'] == '02']
        alaska_counties_fill = filtered_counties[filtered_counties['STATEFP'] == '02']
        alaska_counties_all = all_counties[all_counties['STATEFP'] == '02']
    
        remaining_states = filtered_state_shape_north_america[filtered_state_shape_north_america['STATEFP'] != '02']
        remaining_counties_fill = filtered_counties[filtered_counties['STATEFP'] != '02']
        remaining_counties_all = all_counties[all_counties['STATEFP'] != '02']
    
        # --- Create main plot ---
        fig, ax = plt.subplots(figsize=(15, 7))
    
        if not remaining_counties_fill.empty:
            remaining_counties_fill.plot(
                facecolor=remaining_counties_fill['color'],
                linewidth=0.4,
                ax=ax,
                edgecolor='none',
                legend=False
            )
    
        remaining_counties_all.plot(
            facecolor='none',
            linewidth=0.4,
            ax=ax,
            edgecolor='grey',
            legend=False
        )
    
        remaining_states.plot(
            ax=ax,
            color='none',
            edgecolor='black',
            linewidth=0.8
        )
    
        # --- Alaska inset ---
        # Define clipping polygon (to limit shown extent only)
        polygon = Polygon([(-180, 50), (-180, 72), (-140, 72), (-140, 50)])
        alaska_state_clipped = alaska_state.clip(polygon)
        alaska_counties_fill = alaska_counties_fill.clip(polygon)
        alaska_counties_all = alaska_counties_all.clip(polygon)
    
        # Fixed inset position and size
        ak_ax = fig.add_axes(alaska_inset_position)
    
        if not alaska_counties_fill.empty:
            alaska_counties_fill.plot(
                facecolor=alaska_counties_fill['color'],
                linewidth=0.4,
                ax=ak_ax,
                edgecolor='none',
                legend=False
            )
    
        if not alaska_counties_all.empty:
            alaska_counties_all.plot(
                facecolor='none',
                linewidth=0.4,
                ax=ak_ax,
                edgecolor='grey',
                legend=False
            )
    
        alaska_state_clipped.plot(
            ax=ak_ax,
            color='none',
            edgecolor='black',
            linewidth=0.8
        )
    
        # Fix display area for Alaska inset to prevent auto-scaling
        ak_ax.set_xlim(-180, -140)
        ak_ax.set_ylim(50, 72)
        ak_ax.axis('off')
    
        # Turn off main plot axes
        ax.axis('off')
    
        # --- Create legend ---
        unique_clusters = filtered_counties.groupby('cluster_id')['rank'].min().sort_values().index.tolist()
        cluster_counts = filtered_counties['cluster_id'].value_counts()
        legend_elements = []
    
        for cluster in unique_clusters:
            if cluster == noise_cluster_id:
                count = cluster_counts.get(noise_cluster_id, 0)
                legend_elements.append(Patch(facecolor='blue', label=f'Top noise counties ({count})'))
            else:
                min_rank = filtered_counties.loc[filtered_counties['cluster_id'] == cluster, 'rank'].min()
                color = filtered_counties.loc[filtered_counties['cluster_id'] == cluster, 'color'].values[0]
                count = cluster_counts.get(cluster, 0)
                legend_elements.append(Patch(facecolor=color, label=f'{cluster} (Rank {min_rank}, {count})'))
    
        ax.legend(
            handles=legend_elements,
            title='Cluster ID',
            loc='lower left',
            title_fontsize=10,
            fontsize=8,
            ncol=3  # 3 columns for compact layout
        )
    
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance_heatmap(
        df_scaled,
        df_original_features,
        output_file="feature_importance_heatmap.png",
        figsize=(10, 8),
        dpi=600,
        cmap="YlGnBu"
    ):
        """
        Plots a heatmap of feature importances per cluster using a one-vs-all Random Forest approach.
    
        Parameters:
        - df_scaled: DataFrame containing scaled features and 'cluster_id', 'rank', 'FIPS', 'color', etc.
        - df_original_features: DataFrame with original (unscaled) feature columns to generate clean feature labels.
        - output_file: Filename to save the heatmap image.
        - figsize: Tuple defining figure size.
        - dpi: Resolution of the output image.
        - cmap: Colormap for the heatmap.
        """
    
        # Set seaborn style
        sns.set_context("paper", font_scale=1.2)
        sns.set_style("whitegrid")
    
        # Select feature columns (drop first and last 4 assumed metadata columns)
        X = df_scaled[df_scaled.columns[1:-4]]
        clusters = sorted(df_scaled['cluster_id'].unique())
    
        # Compute feature importances per cluster (one-vs-all strategy)
        feature_importances = [
            {"Cluster": cluster, "Feature": feature, "Importance": int(importance * 100)}
            for cluster in clusters
            for feature, importance in zip(
                X.columns,
                RandomForestClassifier(random_state=42).fit(
                    X, (df_scaled['cluster_id'] == cluster).astype(int)
                ).feature_importances_
            )
        ]
    
        importance_df = pd.DataFrame(feature_importances)
    
        # Pivot to wide format for heatmap
        pivot = importance_df.pivot(index="Feature", columns="Cluster", values="Importance")
    
        # Sort clusters by average rank
        cluster_order = df_scaled.groupby('cluster_id')['rank'].mean().sort_values().index
        pivot = pivot[cluster_order]
    
        # Map original feature names to F1, F2, ...
        feature_id_dict = {feature: f"F{i+1}" for i, feature in enumerate(df_original_features.columns)}
        pivot.index = pivot.index.map(lambda x: feature_id_dict.get(x, x))
    
        # Sort features in reverse numeric order (F10, F9, ..., F1)
        pivot = pivot.sort_index(key=lambda x: x.str.extract(r'(\d+)').astype(int)[0], ascending=False)
    
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(
            pivot, cmap=cmap, annot=True, fmt="d",
            annot_kws={"size": 10}, ax=ax
        )
    
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
    
        # Save and show
        plt.savefig(output_file, format="png", dpi=dpi)
        plt.show()

    @staticmethod
    def plot_cluster_feature_bars(
        df_original,
        df_scaled,
        selected_clusters,
        occupations,
        naics,
        output_file='cluster_feature_values.png'
    ):
        """
        Plots horizontal bar charts showing mean and standard deviation of selected occupation and industry features
        for specified clusters.
    
        Parameters:
        - df_original: original unscaled dataframe
        - df_scaled: scaled dataframe containing 'cluster_id', 'color', and numeric features
        - selected_clusters: list of cluster IDs to visualize
        - occupations: dict mapping occupation codes to readable labels
        - naics: dict mapping NAICS codes to readable labels
        - output_file: filename to save the output plot
        """
    
        # Add cluster info to original DataFrame
        df_original['cluster_id'] = df_scaled['cluster_id']
        df_original['color'] = df_scaled['color']
    
        # Plot style
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
        # Extract lists of relevant codes
        naics_keys = list(naics.keys())
        occ_keys = list(occupations.keys())
    
        # Drop first and last column
        df = df_original.iloc[:, 1:-1]
    
        # Identify relevant columns
        naics_cols = [col for col in df.columns if "naics" in col]
        occ_cols = [col for col in df.columns if "occ" in col]
    
        def extract_code(col_name):
            match = re.search(r'(\d{4,}A?\d*|Automotive|\d{2}-\d{4})$', col_name)
            return match.group() if match else None
    
        # Create mapping: column → code
        occ_mapping = {col: extract_code(col) for col in occ_cols}
        naics_mapping = {col: extract_code(col) for col in naics_cols}
    
        # Filter relevant columns
        occ_cols_filtered = [col for col, code in occ_mapping.items() if code in occ_keys]
        naics_cols_filtered = [col for col, code in naics_mapping.items() if code in naics_keys]
    
        # Create labels
        occ_labels = [occupations[occ_mapping[col]] for col in occ_cols_filtered]
        naics_labels = [naics[naics_mapping[col]] for col in naics_cols_filtered]
    
        def wrap_label(label, width=19):
            wrapped = textwrap.wrap(label, width=width)
            return "\n".join(wrapped[:3])  # max 3 lines
    
        wrapped_occ_labels = [wrap_label(label) for label in occ_labels]
        wrapped_naics_labels = [wrap_label(label) for label in naics_labels]
    
        # Compute cluster-wise means and stds
        cluster_means = df.groupby("cluster_id").mean()
        cluster_stds = df.groupby("cluster_id").std()
    
        # Assign colors to clusters
        cluster_colors = {
            cluster: df_original.loc[df_original["cluster_id"] == cluster, "color"].iloc[0]
            for cluster in selected_clusters
        }
    
        # Layout parameters
        bar_width = 0.2
        n_clusters = len(selected_clusters)
        group_gap = 0.3
    
        indices_occ = np.arange(len(wrapped_occ_labels)) * (n_clusters * bar_width + group_gap)
        y_ticks_occ = indices_occ + (n_clusters * bar_width) / 2
    
        indices_naics = np.arange(len(wrapped_naics_labels)) * (n_clusters * bar_width + group_gap)
        y_ticks_naics = indices_naics + (n_clusters * bar_width) / 2
    
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        error_kw = {'elinewidth': 0.5, 'capsize': 3}
    
        # OCC Plot
        if occ_cols_filtered:
            for i, cluster in enumerate(selected_clusters):
                means = cluster_means.loc[cluster, occ_cols_filtered]
                stds = cluster_stds.loc[cluster, occ_cols_filtered]
                positions = indices_occ + i * bar_width
                axes[0].barh(
                    positions, means.values,
                    height=bar_width,
                    xerr=stds.values,
                    label=f"Cl. {cluster}",
                    error_kw=error_kw,
                    color=cluster_colors[cluster]
                )
            axes[0].set_yticks(y_ticks_occ)
            axes[0].set_yticklabels(wrapped_occ_labels, ha='left')
            axes[0].set_title("Occupation features")
            axes[0].set_xlabel("mean value")
            axes[0].set_xlim(0, 3000)
            axes[0].tick_params(axis='y', pad=105)
            axes[0].legend(loc='upper right', frameon=False)
    
        # NAICS Plot
        if naics_cols_filtered:
            for i, cluster in enumerate(selected_clusters):
                means = cluster_means.loc[cluster, naics_cols_filtered]
                stds = cluster_stds.loc[cluster, naics_cols_filtered]
                positions = indices_naics + i * bar_width
                axes[1].barh(
                    positions, means.values,
                    height=bar_width,
                    xerr=stds.values,
                    label=f"Cl. {cluster}",
                    error_kw=error_kw,
                    color=cluster_colors[cluster]
                )
            axes[1].set_yticks(y_ticks_naics)
            axes[1].set_yticklabels(wrapped_naics_labels, ha='left')
            axes[1].set_title("Industry features")
            axes[1].set_xlabel("mean value")
            axes[1].set_xlim(0, 2000)
            axes[1].tick_params(axis='y', pad=90)
    
        plt.tight_layout(w_pad=0.0)
        plt.savefig(output_file, dpi=600)
        plt.show()



    