"""
Analysis Agent for extracting financial metrics from parsed text
"""
import json
import re
import logging
from openai import OpenAI
from crewai import Agent, Task
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.ULTRASAFE_API_KEY,
            base_url=settings.ULTRASAFE_BASE_URL
        )
        self.model = settings.ULTRASAFE_MODEL
        
    def extract_financial_metrics(self, text: str, context: str = None) -> dict:
        try:
            prompt = self._create_analysis_prompt(text, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst extracting key metrics from financial documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            analysis_result = response.choices[0].message.content
            metrics = self._parse_analysis_response(analysis_result)
            
            logger.info(f"Extracted {len(metrics)} financial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            raise
    
    def _create_analysis_prompt(self, text: str, context: str = None) -> str:
        base_prompt = f"""
        Analyze this financial document and extract key metrics in JSON format:
        
        Document Text:
        {text[:4000]}
        
        Extract this information:
        {{
            "revenue": {{
                "current_year": "value",
                "previous_year": "value",
                "growth_rate": "percentage"
            }},
            "profit_metrics": {{
                "net_income": "value",
                "gross_profit": "value",
                "profit_margin": "percentage"
            }},
            "expenses": {{
                "total_expenses": "value",
                "operating_expenses": "value"
            }},
            "key_ratios": {{
                "debt_to_equity": "ratio",
                "current_ratio": "ratio",
                "return_on_equity": "percentage",
                "earnings_per_share": "value"
            }},
            "year_over_year_changes": {{
                "revenue_change": "percentage",
                "profit_change": "percentage"
            }},
            "notable_trends": ["trend1", "trend2"],
            "risk_factors": ["risk1", "risk2"],
            "company_info": {{
                "company_name": "name",
                "reporting_period": "period",
                "currency": "currency"
            }}
        }}
        
        Use exact figures when available, calculate ratios when possible.
        If a metric is not available, use "N/A".
        """
        
        if context:
            base_prompt += f"\n\nAdditional Context:\n{context[:1000]}"
        
        return base_prompt
    
    def _parse_analysis_response(self, response: str) -> dict:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                metrics = json.loads(json_str)
                return metrics
            else:
                return self._extract_metrics_from_text(response)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, falling back to text extraction")
            return self._extract_metrics_from_text(response)
    
    def _extract_metrics_from_text(self, text: str) -> dict:
        metrics = {
            "revenue": {},
            "profit_metrics": {},
            "expenses": {},
            "key_ratios": {},
            "year_over_year_changes": {},
            "notable_trends": [],
            "risk_factors": [],
            "company_info": {}
        }
        
        # Extract revenue figures
        revenue_pattern = r'revenue[:\s]*\$?([0-9,]+\.?[0-9]*)'
        revenue_match = re.search(revenue_pattern, text.lower())
        if revenue_match:
            metrics["revenue"]["current_year"] = revenue_match.group(1)
        
        # Extract profit figures
        profit_pattern = r'(?:net income|profit)[:\s]*\$?([0-9,]+\.?[0-9]*)'
        profit_match = re.search(profit_pattern, text.lower())
        if profit_match:
            metrics["profit_metrics"]["net_income"] = profit_match.group(1)
        
        # Extract company name
        company_pattern = r'(?:company|corporation|inc|corp)[:\s]*([A-Za-z\s]+)'
        company_match = re.search(company_pattern, text.lower())
        if company_match:
            metrics["company_info"]["company_name"] = company_match.group(1).strip()
        
        return metrics
    
    def identify_trends(self, metrics: dict) -> list:
        trends = []
        
        # Revenue trends
        if "revenue" in metrics:
            revenue_data = metrics["revenue"]
            if "growth_rate" in revenue_data:
                growth_rate = revenue_data["growth_rate"]
                if growth_rate != "N/A":
                    try:
                        rate = float(growth_rate.replace("%", ""))
                        if rate > 10:
                            trends.append("Strong revenue growth")
                        elif rate < 0:
                            trends.append("Revenue decline")
                    except:
                        pass
        
        # Profit trends
        if "profit_metrics" in metrics:
            profit_data = metrics["profit_metrics"]
            if "profit_margin" in profit_data:
                margin = profit_data["profit_margin"]
                if margin != "N/A":
                    try:
                        margin_val = float(margin.replace("%", ""))
                        if margin_val > 20:
                            trends.append("High profit margin")
                        elif margin_val < 5:
                            trends.append("Low profit margin")
                    except:
                        pass
        
        return trends

class FinancialAnalysisAgent:
    def __init__(self):
        self.analyzer = FinancialAnalyzer()
        
    def create_agent(self) -> Agent:
        return Agent(
            role='Financial Analyst',
            goal='Extract and analyze financial metrics from documents',
            backstory="""You are a skilled financial analyst with expertise in 
            extracting key performance indicators from financial documents.""",
            verbose=True,
            allow_delegation=False
        )
    
    def create_task(self, document_data: dict, context: str = None) -> Task:
        return Task(
            description=f"Analyze financial document and extract key metrics",
            agent=self.create_agent(),
            expected_output="Comprehensive financial analysis with metrics and insights"
        )
    
    def execute(self, document_data: dict, context: str = None) -> dict:
        try:
            text = document_data.get('cleaned_text', '')
            
            # Extract financial metrics
            metrics = self.analyzer.extract_financial_metrics(text, context)
            
            # Identify trends
            trends = self.analyzer.identify_trends(metrics)
            if trends:
                metrics["notable_trends"] = trends
            
            # Calculate additional metrics if possible
            if "revenue" in metrics and "profit_metrics" in metrics:
                try:
                    revenue_val = float(metrics["revenue"].get("current_year", "0").replace(",", ""))
                    profit_val = float(metrics["profit_metrics"].get("net_income", "0").replace(",", ""))
                    
                    if revenue_val > 0:
                        profit_margin = (profit_val / revenue_val) * 100
                        metrics["profit_metrics"]["calculated_margin"] = f"{profit_margin:.2f}%"
                except:
                    pass
            
            return {
                "metrics": metrics,
                "analysis_confidence": "high" if len(metrics) > 3 else "medium",
                "extracted_fields": len(metrics),
                "trends_identified": len(trends)
            }
            
        except Exception as e:
            logger.error(f"Error in financial analysis execution: {e}")
            raise 