"""
Report generator for financial summaries
"""
import json
import logging
from datetime import datetime
from openai import OpenAI
from crewai import Agent, Task
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.ULTRASAFE_API_KEY,
            base_url=settings.ULTRASAFE_BASE_URL
        )
        self.model = settings.ULTRASAFE_MODEL
        
    def generate_executive_summary(self, analysis_results: dict) -> dict:
        try:
            metrics = analysis_results.get('metrics', {})
            trends = analysis_results.get('trends', [])
            
            prompt = self._create_report_prompt(metrics, trends)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Create an executive summary from financial data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            report_content = response.choices[0].message.content
            report = self._structure_report(report_content, metrics, trends)
            
            logger.info("Generated report")
            return report
            
        except Exception as e:
            logger.error(f"Report error: {e}")
            raise
    
    def _create_report_prompt(self, metrics: dict, trends: list) -> str:
        company_name = metrics.get('company_info', {}).get('company_name', 'The Company')
        
        prompt = f"""
        Create an executive summary for {company_name}'s financial performance.
        
        Financial Data:
        {json.dumps(metrics, indent=2)}
        
        Trends:
        {json.dumps(trends, indent=2)}
        
        Include:
        1. Executive Summary (2-3 paragraphs)
        2. Key Financial Highlights  
        3. Performance Analysis
        4. Recommendations
        
        Keep it concise and focused on actionable insights.
        """
        
        return prompt
    
    def _structure_report(self, content: str, metrics: dict, trends: list) -> dict:
        company_info = metrics.get('company_info', {})
        company_name = company_info.get('company_name', 'Unknown Company')
        
        structured_report = {
            "report_metadata": {
                "company_name": company_name,
                "report_date": datetime.now().isoformat(),
                "word_count": len(content.split()) if content else 0
            },
            "executive_summary": content,
            "key_metrics_summary": self._create_metrics_summary(metrics),
            "performance_indicators": self._create_performance_indicators(metrics),
            "risk_factors": metrics.get('risk_factors', []),
            "trends": trends,
            "dashboard_summary": self.create_dashboard_summary(metrics)
        }
        
        return structured_report
    
    def _create_metrics_summary(self, metrics: dict) -> dict:
        summary = {}
        
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            summary['revenue'] = {
                'current': revenue.get('current_year', 'N/A'),
                'growth': revenue.get('growth_rate', 'N/A')
            }
        
        if 'profit_metrics' in metrics:
            profit = metrics['profit_metrics']
            summary['profit'] = {
                'net_income': profit.get('net_income', 'N/A'),
                'margin': profit.get('profit_margin', 'N/A')
            }
        
        if 'key_ratios' in metrics:
            ratios = metrics['key_ratios']
            summary['key_ratios'] = {
                'roe': ratios.get('return_on_equity', 'N/A'),
                'current_ratio': ratios.get('current_ratio', 'N/A')
            }
        
        return summary
    
    def _create_performance_indicators(self, metrics: dict) -> dict:
        indicators = {}
        
        # Revenue trend
        if 'revenue' in metrics:
            growth = metrics['revenue'].get('growth_rate', 'N/A')
            if growth != 'N/A':
                try:
                    rate = float(growth.replace('%', ''))
                    if rate > 10:
                        indicators['revenue_trend'] = 'Strong Growth'
                    elif rate > 0:
                        indicators['revenue_trend'] = 'Growing'
                    else:
                        indicators['revenue_trend'] = 'Declining'
                except:
                    indicators['revenue_trend'] = 'Unknown'
        
        # Profitability
        if 'profit_metrics' in metrics:
            margin = metrics['profit_metrics'].get('profit_margin', 'N/A')
            if margin != 'N/A':
                try:
                    margin_val = float(margin.replace('%', ''))
                    if margin_val > 20:
                        indicators['profitability'] = 'High'
                    elif margin_val > 10:
                        indicators['profitability'] = 'Good'
                    else:
                        indicators['profitability'] = 'Low'
                except:
                    indicators['profitability'] = 'Unknown'
        
        return indicators
    
    def create_dashboard_summary(self, metrics: dict) -> dict:
        dashboard = {
            'company_name': metrics.get('company_info', {}).get('company_name', 'Unknown'),
            'overall_health': self._assess_overall_health(metrics),
            'key_trends': [],
            'top_risks': metrics.get('risk_factors', [])[:3]
        }
        
        # Add key trends
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            if revenue.get('growth_rate', 'N/A') != 'N/A':
                dashboard['key_trends'].append(f"Revenue growth: {revenue['growth_rate']}")
        
        if 'profit_metrics' in metrics:
            profit = metrics['profit_metrics']
            if profit.get('profit_margin', 'N/A') != 'N/A':
                dashboard['key_trends'].append(f"Profit margin: {profit['profit_margin']}")
        
        return dashboard
    
    def _assess_overall_health(self, metrics: dict) -> str:
        health_indicators = []
        
        # Check revenue growth
        if 'revenue' in metrics:
            growth = metrics['revenue'].get('growth_rate', 'N/A')
            if growth != 'N/A':
                try:
                    rate = float(growth.replace('%', ''))
                    if rate > 10:
                        health_indicators.append('positive')
                    elif rate < 0:
                        health_indicators.append('negative')
                except:
                    pass
        
        # Check profitability
        if 'profit_metrics' in metrics:
            margin = metrics['profit_metrics'].get('profit_margin', 'N/A')
            if margin != 'N/A':
                try:
                    margin_val = float(margin.replace('%', ''))
                    if margin_val > 15:
                        health_indicators.append('positive')
                    elif margin_val < 5:
                        health_indicators.append('negative')
                except:
                    pass
        
        if not health_indicators:
            return 'Unknown'
        
        positive_count = health_indicators.count('positive')
        negative_count = health_indicators.count('negative')
        
        if positive_count > negative_count:
            return 'Good'
        elif negative_count > positive_count:
            return 'Concerning'
        else:
            return 'Mixed'

class ReportGeneratorAgent:
    def __init__(self):
        self.generator = ReportGenerator()
        
    def create_agent(self) -> Agent:
        return Agent(
            role='Report Generator',
            goal='Create executive summaries',
            backstory="Financial report writer",
            verbose=True,
            allow_delegation=False
        )
    
    def create_task(self, analysis_results: dict) -> Task:
        return Task(
            description="Generate executive summary report",
            agent=self.create_agent(),
            expected_output="Executive summary with insights"
        )
    
    def execute(self, analysis_results: dict) -> dict:
        try:
            return self.generator.generate_executive_summary(analysis_results)
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise 