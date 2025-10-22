"""
Visualization suite for positive patterns analysis

Creates rich visualizations including:
- Heatmaps of cognitive actions
- Token trajectory plots
- Layer distribution charts
- Statistical comparisons
- Interactive dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


class VisualizationSuite:
    """Comprehensive visualization suite for analysis results"""

    def __init__(self, output_dir: Path):
        """
        Initialize visualization suite

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11

    def plot_action_heatmap(
        self,
        summary_df: pd.DataFrame,
        top_n: int = 30,
        figsize: tuple = (14, 10)
    ):
        """
        Create heatmap showing cognitive action confidences by statement type

        Args:
            summary_df: Summary statistics DataFrame
            top_n: Number of top actions to show
            figsize: Figure size
        """
        print("Creating action heatmap...")

        # Select top actions by absolute difference
        top_actions = summary_df.nlargest(top_n, 'pos_minus_neg_mean')['action'].tolist()

        # Prepare data
        heatmap_data = []
        for action in top_actions:
            row = summary_df[summary_df['action'] == action].iloc[0]
            heatmap_data.append({
                'Action': action,
                'Positive': row['pos_mean'],
                'Negative': row['neg_mean'],
                'Transformed': row['trans_mean']
            })

        df_heat = pd.DataFrame(heatmap_data).set_index('Action')

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df_heat,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.15,
            vmin=0,
            vmax=0.5,
            cbar_kws={'label': 'Mean Confidence'},
            ax=ax
        )

        ax.set_title(f'Top {top_n} Cognitive Actions by Statement Type', fontsize=16, fontweight='bold')
        ax.set_xlabel('Statement Type', fontsize=13)
        ax.set_ylabel('Cognitive Action', fontsize=13)

        plt.tight_layout()
        output_path = self.figures_dir / 'action_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved heatmap to {output_path}")

    def plot_difference_bars(
        self,
        summary_df: pd.DataFrame,
        top_n: int = 20,
        figsize: tuple = (14, 10)
    ):
        """
        Create bar chart showing biggest positive vs negative differences

        Args:
            summary_df: Summary statistics DataFrame
            top_n: Number of actions to show
            figsize: Figure size
        """
        print("Creating difference bar chart...")

        # Get top positive and negative differences
        top_positive = summary_df.nlargest(top_n // 2, 'pos_minus_neg_mean')
        top_negative = summary_df.nsmallest(top_n // 2, 'pos_minus_neg_mean')

        combined = pd.concat([top_positive, top_negative]).sort_values('pos_minus_neg_mean')

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['#d62728' if x < 0 else '#2ca02c' for x in combined['pos_minus_neg_mean']]

        ax.barh(combined['action'], combined['pos_minus_neg_mean'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        ax.set_xlabel('Confidence Difference (Positive - Negative)', fontsize=13)
        ax.set_ylabel('Cognitive Action', fontsize=13)
        ax.set_title('Actions Differentiating Positive vs Negative Patterns', fontsize=16, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.7, label='More in Positive'),
            Patch(facecolor='#d62728', alpha=0.7, label='More in Negative')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        output_path = self.figures_dir / 'difference_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved difference bars to {output_path}")

    def plot_transformation_flow(
        self,
        summary_df: pd.DataFrame,
        top_n: int = 15,
        figsize: tuple = (14, 8)
    ):
        """
        Visualize how transformation statements bridge from negative to positive

        Args:
            summary_df: Summary statistics DataFrame
            top_n: Number of top actions
            figsize: Figure size
        """
        print("Creating transformation flow chart...")

        # Select actions with highest transformation effect
        top_actions = summary_df.nlargest(top_n, 'trans_minus_neg_mean')

        # Prepare data
        actions = top_actions['action'].tolist()
        neg = top_actions['neg_mean'].tolist()
        trans = top_actions['trans_mean'].tolist()
        pos = top_actions['pos_mean'].tolist()

        x = np.arange(len(actions))
        width = 0.25

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(x - width, neg, width, label='Negative', color='#d62728', alpha=0.7)
        ax.bar(x, trans, width, label='Transformed', color='#ff7f0e', alpha=0.7)
        ax.bar(x + width, pos, width, label='Positive', color='#2ca02c', alpha=0.7)

        # Add arrows showing transformation
        for i in range(len(actions)):
            # Arrow from negative to transformed
            ax.annotate('', xy=(i, trans[i]), xytext=(i - width, neg[i]),
                       arrowprops=dict(arrowstyle='->', lw=1, alpha=0.3, color='gray'))
            # Arrow from transformed to positive
            ax.annotate('', xy=(i + width, pos[i]), xytext=(i, trans[i]),
                       arrowprops=dict(arrowstyle='->', lw=1, alpha=0.3, color='gray'))

        ax.set_xlabel('Cognitive Action', fontsize=13)
        ax.set_ylabel('Mean Confidence', fontsize=13)
        ax.set_title('Transformation Flow: Negative → Transformed → Positive', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.figures_dir / 'transformation_flow.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved transformation flow to {output_path}")

    def plot_pattern_type_analysis(
        self,
        statement_df: pd.DataFrame,
        figsize: tuple = (16, 10)
    ):
        """
        Analyze cognitive actions by cognitive pattern type

        Args:
            statement_df: Statement-level DataFrame
            figsize: Figure size
        """
        print("Creating pattern type analysis...")

        # Group by pattern type and statement type
        grouped = statement_df.groupby(['cognitive_pattern_type', 'statement_type'])['max_confidence'].mean().reset_index()

        # Pivot for plotting
        pivot = grouped.pivot(index='cognitive_pattern_type', columns='statement_type', values='max_confidence')

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        pivot.plot(kind='bar', ax=ax, width=0.8, alpha=0.7)

        ax.set_xlabel('Cognitive Pattern Type', fontsize=13)
        ax.set_ylabel('Mean Confidence', fontsize=13)
        ax.set_title('Cognitive Action Confidence by Pattern Type', fontsize=16, fontweight='bold')
        ax.legend(title='Statement Type', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.figures_dir / 'pattern_type_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved pattern type analysis to {output_path}")

    def plot_layer_distribution(
        self,
        statement_df: pd.DataFrame,
        top_n: int = 15,
        figsize: tuple = (14, 8)
    ):
        """
        Show which layers detect which cognitive actions

        Args:
            statement_df: Statement-level DataFrame
            top_n: Number of top actions
            figsize: Figure size
        """
        print("Creating layer distribution plot...")

        # Get top actions
        top_actions = statement_df.groupby('action')['max_confidence'].mean().nlargest(top_n).index.tolist()

        # Filter to top actions
        df_filtered = statement_df[statement_df['action'].isin(top_actions)]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Violin plot
        sns.violinplot(
            data=df_filtered,
            x='action',
            y='best_layer',
            hue='statement_type',
            split=False,
            ax=ax,
            palette='Set2'
        )

        ax.set_xlabel('Cognitive Action', fontsize=13)
        ax.set_ylabel('Best Layer', fontsize=13)
        ax.set_title('Layer Distribution for Top Cognitive Actions', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Statement Type', fontsize=10)

        plt.tight_layout()
        output_path = self.figures_dir / 'layer_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved layer distribution to {output_path}")

    def plot_confidence_distributions(
        self,
        summary_df: pd.DataFrame,
        top_n: int = 12,
        figsize: tuple = (16, 10)
    ):
        """
        Create box plots showing confidence distributions

        Args:
            summary_df: Summary statistics DataFrame
            top_n: Number of actions to show
            figsize: Figure size
        """
        print("Creating confidence distribution plots...")

        # Get top actions by variance
        top_actions = summary_df.nlargest(top_n, 'pos_minus_neg_mean')['action'].tolist()

        # Prepare data for plotting
        plot_data = []
        for _, row in summary_df[summary_df['action'].isin(top_actions)].iterrows():
            for stmt_type, mean_col, std_col in [
                ('Positive', 'pos_mean', 'pos_std'),
                ('Negative', 'neg_mean', 'neg_std'),
                ('Transformed', 'trans_mean', 'trans_std')
            ]:
                plot_data.append({
                    'Action': row['action'],
                    'Statement Type': stmt_type,
                    'Mean': row[mean_col],
                    'Std': row[std_col]
                })

        df_plot = pd.DataFrame(plot_data)

        # Create plot with error bars
        fig, ax = plt.subplots(figsize=figsize)

        actions_list = top_actions
        x_pos = np.arange(len(actions_list))
        width = 0.25

        for i, stmt_type in enumerate(['Negative', 'Transformed', 'Positive']):
            df_type = df_plot[df_plot['Statement Type'] == stmt_type]
            means = [df_type[df_type['Action'] == a]['Mean'].values[0] for a in actions_list]
            stds = [df_type[df_type['Action'] == a]['Std'].values[0] for a in actions_list]

            color = {'Negative': '#d62728', 'Transformed': '#ff7f0e', 'Positive': '#2ca02c'}[stmt_type]

            ax.bar(x_pos + i * width, means, width, yerr=stds,
                  label=stmt_type, alpha=0.7, color=color,
                  capsize=3)

        ax.set_xlabel('Cognitive Action', fontsize=13)
        ax.set_ylabel('Mean Confidence ± Std Dev', fontsize=13)
        ax.set_title('Confidence Distributions by Statement Type', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(actions_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.figures_dir / 'confidence_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved confidence distributions to {output_path}")

    def create_interactive_dashboard(
        self,
        summary_df: pd.DataFrame,
        statement_df: pd.DataFrame
    ):
        """
        Create interactive Plotly dashboard

        Args:
            summary_df: Summary statistics DataFrame
            statement_df: Statement-level DataFrame
        """
        print("Creating interactive dashboard...")

        # 1. Interactive heatmap
        top_actions = summary_df.nlargest(25, 'pos_minus_neg_mean')['action'].tolist()
        heatmap_data = []
        for action in top_actions:
            row = summary_df[summary_df['action'] == action].iloc[0]
            heatmap_data.append([row['neg_mean'], row['trans_mean'], row['pos_mean']])

        fig1 = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Negative', 'Transformed', 'Positive'],
            y=top_actions,
            colorscale='RdYlGn',
            text=[[f'{v:.3f}' for v in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Confidence")
        ))

        fig1.update_layout(
            title='Cognitive Actions Heatmap',
            xaxis_title='Statement Type',
            yaxis_title='Cognitive Action',
            height=800
        )

        # 2. Interactive scatter plot
        scatter_data = summary_df.head(50).copy()
        scatter_data['abs_diff'] = scatter_data['pos_minus_neg_mean'].abs()

        fig2 = px.scatter(
            scatter_data,
            x='neg_mean',
            y='pos_mean',
            size='abs_diff',
            color='pos_minus_neg_mean',
            hover_data=['action', 'trans_mean'],
            labels={
                'neg_mean': 'Negative Confidence',
                'pos_mean': 'Positive Confidence',
                'pos_minus_neg_mean': 'Difference'
            },
            title='Positive vs Negative Confidence Scatter',
            color_continuous_scale='RdYlGn'
        )

        fig2.add_shape(
            type='line',
            x0=0, y0=0, x1=0.5, y1=0.5,
            line=dict(color='gray', dash='dash')
        )

        # 3. Box plot by pattern type
        fig3 = px.box(
            statement_df,
            x='statement_type',
            y='max_confidence',
            color='cognitive_pattern_type',
            title='Confidence Distribution by Pattern Type',
            labels={
                'statement_type': 'Statement Type',
                'max_confidence': 'Confidence',
                'cognitive_pattern_type': 'Pattern Type'
            }
        )

        # Save interactive figures
        fig1.write_html(self.figures_dir / 'interactive_heatmap.html')
        fig2.write_html(self.figures_dir / 'interactive_scatter.html')
        fig3.write_html(self.figures_dir / 'interactive_boxplot.html')

        print(f"✓ Saved interactive dashboard to {self.figures_dir}")

    def generate_all_visualizations(
        self,
        summary_df: pd.DataFrame,
        statement_df: pd.DataFrame
    ):
        """
        Generate all visualizations

        Args:
            summary_df: Summary statistics DataFrame
            statement_df: Statement-level DataFrame
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        print()

        self.plot_action_heatmap(summary_df)
        self.plot_difference_bars(summary_df)
        self.plot_transformation_flow(summary_df)
        self.plot_pattern_type_analysis(statement_df)
        self.plot_layer_distribution(statement_df)
        self.plot_confidence_distributions(summary_df)
        self.create_interactive_dashboard(summary_df, statement_df)

        print("\n✓ All visualizations generated!")
