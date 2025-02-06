# visualization_agent.py
from typing import Dict, Any
from .base_agent import BaseAgent
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class VisualizationAgent(BaseAgent):
    def __init__(self, registry):
        super().__init__(registry)
        self.role = "Visualization Agent"
        self.goal = (
            "To create clear and informative visualizations from investment analysis reports "
            "using Streamlit and Plotly, focusing on key metrics including confidence levels, "
            "growth indicators, and multi-dimensional analysis through radar charts."
        )
        self.backstory = (
            "As a data visualization specialist with expertise in financial analytics, "
            "this agent transforms complex investment analyses into intuitive visual "
            "representations using modern visualization libraries."
        )
        self.tools = []
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client_OpenAI = OpenAI()
        
        # Define common style elements
        self.color_scheme = {
            'primary': '#3F51B5',      # primary-100
            'secondary': '#757de8',     # primary-200
            'accent': '#2196F3',        # accent-100
            'accent_dark': '#003f8f',   # accent-200
            'background': '#FFFFFF',    # bg-100
            'background_alt': '#f5f5f5',# bg-200
            'grid': '#cccccc',          # bg-300
            'text': '#333333',          # text-100
            'text_light': '#5c5c5c',    # text-200
            'highlight': '#dedeff'      # primary-300
        }
        
        self.common_layout = {
            'font': {
                'family': 'Montserrat, sans-serif',
                'size': 12,
                'color': self.color_scheme['text']
            },
            'plot_bgcolor': self.color_scheme['background'],
            'paper_bgcolor': self.color_scheme['background'],
            'margin': dict(t=40, b=40, l=40, r=40)
        }

    def get_system_prompt(self, system_prompt=None) -> str:
        """Implement required abstract method from BaseAgent"""
        default_prompt = (
            "You are a visualization specialist focused on creating clear and informative "
            "financial visualizations. Your goal is to transform complex investment data "
            "into intuitive visual representations that help stakeholders make informed decisions."
        )
        return system_prompt if system_prompt else default_prompt

    def process_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Implement required abstract method from BaseAgent"""
        return arguments

    def create_confidence_gauge(self, confidence_score: float):
        """Create a confidence gauge visualization using Plotly"""
        colors = [
            self.color_scheme['grid'],      # 0-50
            '#7F8C8D',                      # 50-75
            self.color_scheme['accent']      # 75-100
        ]
        
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "Investment Confidence Score",
                'font': {
                    'size': 24, 
                    'color': 'black',
                    'weight': 'bold'
                }
            },
            number={'font': {'color': self.color_scheme['primary']}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 2,
                    'tickcolor': self.color_scheme['text']
                },
                'bar': {'color': self.color_scheme['primary']},
                'bgcolor': self.color_scheme['background'],
                'borderwidth': 2,
                'bordercolor': self.color_scheme['text'],
                'steps': [
                    {'range': [0, 50], 'color': self.color_scheme['background_alt']},
                    {'range': [50, 75], 'color': self.color_scheme['secondary']},
                    {'range': [75, 100], 'color': self.color_scheme['highlight']}
                ]
            }
        ))

        fig.update_layout(
            **self.common_layout,
            height=400
        )

        return fig

    def create_growth_chart(self, growth_indicators: Dict):
        """Create a growth indicators bar chart using Plotly"""
        df = pd.DataFrame({
            'Indicator': list(growth_indicators.keys()),
            'Value': list(growth_indicators.values())
        })

        fig = go.Figure(data=[
            go.Bar(
                x=df['Indicator'],
                y=df['Value'],
                marker_color=self.color_scheme['primary'],
                text=df['Value'].round(1),
                textposition='auto',
                marker_line_color=self.color_scheme['secondary'],
                marker_line_width=1.5,
                textfont={'color': self.color_scheme['text']}
            )
        ])

        fig.update_layout(
            **self.common_layout,
            title={
                'text': 'Growth Indicators',
                'font': {
                    'size': 24,
                    'color': 'black'
                },
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis={
                'title': None,
                'tickangle': -45,
                'gridcolor': self.color_scheme['grid']
            },
            yaxis={
                'title': 'Value (%)',
                'gridcolor': self.color_scheme['grid'],
                'range': [0, max(df['Value']) * 1.1]
            },
            showlegend=False,
            height=400,
            bargap=0.3
        )

        return fig

    def create_radar_chart(self, radar_metrics: Dict):
        """Create a radar chart using Plotly"""
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba(63, 81, 181, 0.4)',  # Semi-transparent primary color
            line=dict(
                color=self.color_scheme['primary'],
                width=2
            ),
            name='Performance Metrics'
        ))

        fig.update_layout(
            **self.common_layout,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    gridcolor=self.color_scheme['grid']
                ),
                angularaxis=dict(
                    showline=True,
                    linecolor=self.color_scheme['text'],
                    gridcolor=self.color_scheme['grid']
                ),
                bgcolor=self.color_scheme['background']
            ),
            title={
                'text': "Company Performance Analysis",
                'font': {
                    'size': 24,
                    'color': 'black'
                },
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False,
            height=400
        )

        return fig

    def generate_visualizations(self, metrics: Dict) -> Dict:
        """Generate all visualizations using Plotly"""
        return {
            "confidence": self.create_confidence_gauge(metrics["confidence_score"]),
            "growth": self.create_growth_chart(metrics["growth_indicators"]),
            "radar": self.create_radar_chart(metrics["radar_metrics"])
        }

    def extract_metrics_prompt(self, analysis_result: str) -> str:
        """Generate a prompt to extract metrics from the analysis result"""
        prompt = f"""
Please analyze this investment report and extract/estimate the following metrics. Base your estimates on the report content.

REPORT TO ANALYZE:
{analysis_result}

EXTRACTION RULES:
1. All metrics should be scored 0-100
2. Use actual numbers from the report where available
3. For missing metrics, derive reasonable estimates from the context
4. Convert any growth percentages to 0-100 scale appropriately

REQUIRED FORMAT:
{{
    "Confidence Score": <confidence_level_from_report>,
    "Growth Indicators": {{
        "Revenue Growth Rate": <based_on_reported_revenue_growth>,
        "Profit Margin": <based_on_gross_margin>,
        "Market Share Growth": <estimate_from_market_position>,
        "Customer Base Growth": <estimate_from_revenue_growth>,
        "Product Innovation Score": <based_on_ai_gaming_focus>
    }},
    "Radar Chart Metrics": {{
        "Financial Health": <based_on_reported_financials>,
        "Market Position": <based_on_competitive_analysis>,
        "Growth Potential": <based_on_future_outlook>,
        "Risk Level": <based_on_risk_assessment>,
        "Management Quality": <based_on_strategic_decisions>,
        "Innovation Capacity": <based_on_product_development>
    }}
}}
"""
        return prompt

    def extract_metrics(self, analysis_result: str) -> Dict:
        """Extract metrics from the analysis result using LLM"""
        prompt = self.extract_metrics_prompt(analysis_result)
        
        try:
            response = self.client_OpenAI.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a data extraction specialist."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            
            content = response.choices[0].message.content
            print("LLM Response:", content)
            
            try:
                metrics = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'{.*}', content, re.DOTALL)
                if json_match:
                    metrics = json.loads(json_match.group())
                else:
                    raise Exception("Could not extract JSON from response")

            # Validate and format the metrics
            validated_metrics = self.validate_metrics(metrics)
            return validated_metrics

        except Exception as e:
            print(f"Error in metric extraction: {str(e)}")
            return self.validate_metrics({})

    def validate_metrics(self, metrics: Dict) -> Dict:
        """Validate and format the metrics dictionary"""
        expected_structure = {
            "confidence_score": 0,
            "growth_indicators": {
                "revenue_growth": 0,
                "profit_margin": 0,
                "market_share_growth": 0,
                "customer_base_growth": 0,
                "product_innovation": 0
            },
            "radar_metrics": {
                "financial_health": 0,
                "market_position": 0,
                "growth_potential": 0,
                "risk_level": 0,
                "management_quality": 0,
                "innovation_capacity": 0
            }
        }

        validated_metrics = expected_structure.copy()

        # Helper function to normalize keys
        def normalize_key(key: str) -> str:
            return key.lower().replace(" ", "_")

        # Extract confidence score
        confidence_keys = ["confidence_score", "confidence score", "Confidence Score", "confidence"]
        for key in confidence_keys:
            if key in metrics and metrics[key] is not None:
                validated_metrics["confidence_score"] = float(metrics[key])
                break

        # Extract growth indicators
        growth_data = metrics.get("Growth Indicators") or metrics.get("growth_indicators") or {}
        for key, value in growth_data.items():
            normalized_key = normalize_key(key)
            if normalized_key in validated_metrics["growth_indicators"] and value is not None:
                validated_metrics["growth_indicators"][normalized_key] = float(value)

        # Extract radar metrics
        radar_data = metrics.get("Radar Chart Metrics") or metrics.get("radar_metrics") or {}
        for key, value in radar_data.items():
            normalized_key = normalize_key(key)
            if normalized_key in validated_metrics["radar_metrics"] and value is not None:
                validated_metrics["radar_metrics"][normalized_key] = float(value)

        # Replace None values with default values
        for category in ["growth_indicators", "radar_metrics"]:
            for key in validated_metrics[category]:
                if validated_metrics[category][key] is None:
                    validated_metrics[category][key] = 0.0

        print("Validated metrics:", validated_metrics)
        return validated_metrics

    def run(self, analysis_result: str) -> Dict:
        """Main execution method to process analysis result and generate visualizations"""
        try:
            print("Extracting metrics from analysis...")
            metrics = self.extract_metrics(analysis_result)
            print("Extracted metrics:", metrics)
            
            print("Validating metrics...")
            validated_metrics = self.validate_metrics(metrics)
            print("Validated metrics:", validated_metrics)
            
            print("Generating visualizations...")
            visualizations = self.generate_visualizations(validated_metrics)
            print("Visualizations generated successfully")
            
            return visualizations
        except Exception as e:
            print(f"Error in visualization generation: {str(e)}")
            raise