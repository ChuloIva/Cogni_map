"""
HTML report generator for positive patterns analysis

Creates comprehensive HTML reports with:
- Executive summary
- Key findings
- Embedded visualizations
- Statistical tables
- Sample analyses
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import base64


class HTMLReportGenerator:
    """Generate comprehensive HTML reports"""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"

    def _image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 for embedding"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def _create_css(self) -> str:
        """Create CSS styles for report"""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 5px 0;
                font-size: 1.1em;
                opacity: 0.9;
            }
            .section {
                background: white;
                padding: 30px;
                margin-bottom: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .section h3 {
                color: #764ba2;
                margin-top: 25px;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .metric-card h3 {
                margin: 0 0 10px 0;
                color: #667eea;
                font-size: 1.1em;
            }
            .metric-card .value {
                font-size: 2.5em;
                font-weight: bold;
                color: #333;
            }
            .metric-card .label {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
            }
            th {
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .positive { color: #2ca02c; font-weight: bold; }
            .negative { color: #d62728; font-weight: bold; }
            .neutral { color: #ff7f0e; font-weight: bold; }
            .finding {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px 20px;
                margin: 15px 0;
                border-radius: 4px;
            }
            .finding strong {
                color: #667eea;
            }
            .visualization {
                margin: 30px 0;
                text-align: center;
            }
            .visualization img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .visualization iframe {
                width: 100%;
                height: 600px;
                border: none;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .footer {
                text-align: center;
                color: #666;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .toc {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .toc h3 {
                margin-top: 0;
                color: #667eea;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 0;
            }
            .toc li {
                padding: 5px 0;
            }
            .toc a {
                color: #667eea;
                text-decoration: none;
            }
            .toc a:hover {
                text-decoration: underline;
            }
        </style>
        """

    def generate_report(
        self,
        summary_df: pd.DataFrame,
        statement_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        num_entries: int,
        sample_size: Optional[int] = None
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            summary_df: Summary statistics DataFrame
            statement_df: Statement-level DataFrame
            comparison_df: Entry comparison DataFrame
            num_entries: Total number of entries analyzed
            sample_size: Sample size if applicable

        Returns:
            Path to generated HTML file
        """
        print("Generating HTML report...")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate key metrics
        total_actions = len(summary_df)
        avg_pos_conf = summary_df['pos_mean'].mean()
        avg_neg_conf = summary_df['neg_mean'].mean()
        avg_trans_conf = summary_df['trans_mean'].mean()

        # Top positive and negative actions
        top_positive = summary_df.nlargest(10, 'pos_minus_neg_mean')
        top_negative = summary_df.nsmallest(10, 'pos_minus_neg_mean')
        top_transformation = summary_df.nlargest(10, 'trans_minus_neg_mean')

        # Pattern type breakdown
        pattern_types = statement_df['cognitive_pattern_type'].unique()

        # Start HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Positive Patterns Analysis Report</title>
            {self._create_css()}
        </head>
        <body>
            <div class="header">
                <h1>Positive Patterns Analysis Report</h1>
                <p>Comprehensive evaluation of cognitive action patterns in healthy vs unhealthy thought processes</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Entries Analyzed:</strong> {num_entries} {f'(sample of {sample_size})' if sample_size else '(full dataset)'}</p>
            </div>

            <div class="toc">
                <h3>Table of Contents</h3>
                <ul>
                    <li><a href="#overview">1. Executive Overview</a></li>
                    <li><a href="#findings">2. Key Findings</a></li>
                    <li><a href="#positive">3. Positive Pattern Actions</a></li>
                    <li><a href="#negative">4. Negative Pattern Actions</a></li>
                    <li><a href="#transformation">5. Transformation Analysis</a></li>
                    <li><a href="#patterns">6. Pattern Type Analysis</a></li>
                    <li><a href="#visualizations">7. Visualizations</a></li>
                </ul>
            </div>

            <div id="overview" class="section">
                <h2>1. Executive Overview</h2>

                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Total Actions</h3>
                        <div class="value">{total_actions}</div>
                        <div class="label">Cognitive actions detected</div>
                    </div>
                    <div class="metric-card">
                        <h3>Positive Avg</h3>
                        <div class="value">{avg_pos_conf:.3f}</div>
                        <div class="label">Mean confidence</div>
                    </div>
                    <div class="metric-card">
                        <h3>Negative Avg</h3>
                        <div class="value">{avg_neg_conf:.3f}</div>
                        <div class="label">Mean confidence</div>
                    </div>
                    <div class="metric-card">
                        <h3>Transformed Avg</h3>
                        <div class="value">{avg_trans_conf:.3f}</div>
                        <div class="label">Mean confidence</div>
                    </div>
                </div>

                <p>
                    This analysis examined <strong>{num_entries}</strong> thought pattern entries across
                    <strong>{len(pattern_types)}</strong> different cognitive pattern types. Using dual probe
                    inference (token-by-token and whole-string analysis), we identified which cognitive actions
                    differentiate healthy (positive) from unhealthy (negative) thought patterns, and how
                    transformation statements facilitate positive change.
                </p>
            </div>

            <div id="findings" class="section">
                <h2>2. Key Findings</h2>

                <div class="finding">
                    <strong>Finding 1:</strong> Positive thought patterns show
                    <span class="positive">{(avg_pos_conf - avg_neg_conf):.3f}</span> higher average confidence
                    in cognitive actions compared to negative patterns, suggesting more active cognitive engagement.
                </div>

                <div class="finding">
                    <strong>Finding 2:</strong> Transformation statements bridge the gap between negative and
                    positive patterns with an average confidence of <span class="neutral">{avg_trans_conf:.3f}</span>,
                    demonstrating their role in facilitating cognitive shifts.
                </div>

                <div class="finding">
                    <strong>Finding 3:</strong> The top cognitive actions distinguishing positive from negative
                    patterns include: <strong>{', '.join(top_positive.head(5)['action'].tolist())}</strong>.
                </div>

                <div class="finding">
                    <strong>Finding 4:</strong> Negative patterns are characterized by:
                    <strong>{', '.join(top_negative.head(5)['action'].tolist())}</strong>.
                </div>
            </div>

            <div id="positive" class="section">
                <h2>3. Positive Pattern Actions</h2>
                <p>These cognitive actions are most strongly associated with healthy, positive thought patterns:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Cognitive Action</th>
                            <th>Positive</th>
                            <th>Negative</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, (_, row) in enumerate(top_positive.iterrows(), 1):
            diff_class = "positive" if row['pos_minus_neg_mean'] > 0 else "negative"
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><strong>{row['action']}</strong></td>
                            <td>{row['pos_mean']:.4f}</td>
                            <td>{row['neg_mean']:.4f}</td>
                            <td class="{diff_class}">{row['pos_minus_neg_mean']:+.4f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>

            <div id="negative" class="section">
                <h2>4. Negative Pattern Actions</h2>
                <p>These cognitive actions are most strongly associated with unhealthy, negative thought patterns:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Cognitive Action</th>
                            <th>Negative</th>
                            <th>Positive</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, (_, row) in enumerate(top_negative.iterrows(), 1):
            diff_class = "negative" if row['pos_minus_neg_mean'] < 0 else "positive"
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><strong>{row['action']}</strong></td>
                            <td>{row['neg_mean']:.4f}</td>
                            <td>{row['pos_mean']:.4f}</td>
                            <td class="{diff_class}">{row['pos_minus_neg_mean']:+.4f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>

            <div id="transformation" class="section">
                <h2>5. Transformation Analysis</h2>
                <p>These cognitive actions show the strongest transformation effect (negative → transformed):</p>

                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Cognitive Action</th>
                            <th>Negative</th>
                            <th>Transformed</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, (_, row) in enumerate(top_transformation.iterrows(), 1):
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><strong>{row['action']}</strong></td>
                            <td>{row['neg_mean']:.4f}</td>
                            <td>{row['trans_mean']:.4f}</td>
                            <td class="positive">{row['trans_minus_neg_mean']:+.4f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>

            <div id="patterns" class="section">
                <h2>6. Pattern Type Analysis</h2>
                <p>Breakdown by cognitive pattern type:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Pattern Type</th>
                            <th>Count</th>
                            <th>Avg Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        pattern_stats = statement_df.groupby('cognitive_pattern_type').agg({
            'entry_id': 'count',
            'max_confidence': 'mean'
        }).reset_index()
        pattern_stats.columns = ['pattern_type', 'count', 'avg_confidence']
        pattern_stats = pattern_stats.sort_values('avg_confidence', ascending=False)

        for _, row in pattern_stats.iterrows():
            html += f"""
                        <tr>
                            <td><strong>{row['pattern_type']}</strong></td>
                            <td>{row['count']}</td>
                            <td>{row['avg_confidence']:.4f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>

            <div id="visualizations" class="section">
                <h2>7. Visualizations</h2>

                <h3>7.1 Action Heatmap</h3>
                <div class="visualization">
        """

        # Embed static images
        if (self.figures_dir / 'action_heatmap.png').exists():
            html += f'<img src="figures/action_heatmap.png" alt="Action Heatmap">'

        html += """
                </div>

                <h3>7.2 Difference Analysis</h3>
                <div class="visualization">
        """

        if (self.figures_dir / 'difference_bars.png').exists():
            html += f'<img src="figures/difference_bars.png" alt="Difference Bars">'

        html += """
                </div>

                <h3>7.3 Transformation Flow</h3>
                <div class="visualization">
        """

        if (self.figures_dir / 'transformation_flow.png').exists():
            html += f'<img src="figures/transformation_flow.png" alt="Transformation Flow">'

        html += """
                </div>

                <h3>7.4 Confidence Distributions</h3>
                <div class="visualization">
        """

        if (self.figures_dir / 'confidence_distributions.png').exists():
            html += f'<img src="figures/confidence_distributions.png" alt="Confidence Distributions">'

        html += """
                </div>

                <h3>7.5 Interactive Heatmap</h3>
                <div class="visualization">
        """

        if (self.figures_dir / 'interactive_heatmap.html').exists():
            html += '<iframe src="figures/interactive_heatmap.html"></iframe>'

        html += """
                </div>

                <h3>7.6 Interactive Scatter Plot</h3>
                <div class="visualization">
        """

        if (self.figures_dir / 'interactive_scatter.html').exists():
            html += '<iframe src="figures/interactive_scatter.html"></iframe>'

        html += """
                </div>
            </div>

            <div class="footer">
                <p>Generated by Positive Patterns Evaluator</p>
                <p>Using StreamingProbeInferenceEngine and UniversalMultiLayerInferenceEngine</p>
            </div>
        </body>
        </html>
        """

        # Save report
        output_path = self.output_dir / "analysis_report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ Saved HTML report to {output_path}")

        return str(output_path)
