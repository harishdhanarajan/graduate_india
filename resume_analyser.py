# resume_analyzer_modular.py
"""
Modular Resume Analyzer
Breaks down analysis into separate, focused components
"""

import os
import json
import hashlib
import re
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Rich library imports
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize console globally
console = Console()

# ===== CONFIGURATION =====
@dataclass
class Config:
    """Central configuration for the analyzer"""
    # OpenAI Settings
    MODEL = "gpt-4-1106-preview"
    FAST_MODEL = "gpt-3.5-turbo"  # For simpler tasks
    TEMPERATURE = 0.1  # Low for consistency
    MAX_TOKENS = 2048  # Reduced since we're making focused queries
    
    # Application Settings
    ANALYSIS_HISTORY_DIR = "analyses"
    RESUME_FILE_PATH = "resume.txt"
    
    # Scoring Thresholds
    EXCELLENT_THRESHOLD = 85
    GOOD_THRESHOLD = 70
    
    # Technical Keywords Database
    TECHNICAL_KEYWORDS = {
        'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'scala', 'r', 'matlab'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'dynamodb', 'cassandra', 'oracle'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
        'aws_services': ['ec2', 's3', 'lambda', 'glue', 'emr', 'rds', 'redshift', 'sqs', 'sns', 'cloudwatch'],
        'data_tools': ['spark', 'hadoop', 'kafka', 'airflow', 'databricks', 'tableau', 'power bi'],
        'ml_ai': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'nlp', 'deep learning', 'machine learning'],
        'web': ['react', 'angular', 'vue', 'django', 'flask', 'rest api', 'graphql'],
        'devops': ['ci/cd', 'git', 'agile', 'scrum', 'devops']
    }


# ===== MODULE 1: KEYWORD EXTRACTION (PROGRAMMATIC) =====
class KeywordAnalyzer:
    """Handles technical keyword extraction and counting"""
    
    def __init__(self):
        # Flatten all keywords into a single set for searching
        self.all_keywords = set()
        for category, keywords in Config.TECHNICAL_KEYWORDS.items():
            self.all_keywords.update(keywords)
    
    def extract_keywords(self, resume_text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract and count technical keywords from resume"""
        resume_lower = resume_text.lower()
        keyword_counts = {}
        
        # Count exact matches for known technical keywords
        for keyword in self.all_keywords:
            # Use word boundaries for exact matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, resume_lower)
            if matches:
                keyword_counts[keyword.title()] = len(matches)
        
        # Also find technical-looking terms not in our list
        # Match: AWS, S3, CI/CD, PySpark, etc.
        tech_pattern = r'\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][a-z]*|[a-z]+[0-9]+|[A-Z]+[0-9]+)\b'
        potential_terms = re.findall(tech_pattern, resume_text)
        
        # Count these terms
        term_counts = Counter(potential_terms)
        
        # Add significant terms not already counted
        for term, count in term_counts.items():
            if term not in keyword_counts and count >= 2 and len(term) > 1:
                keyword_counts[term] = count
        
        # Sort and return top N
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"term": term, "count": count} for term, count in sorted_keywords]
    
    def get_keyword_categories(self, resume_text: str) -> Dict[str, List[str]]:
        """Categorize found keywords by type"""
        resume_lower = resume_text.lower()
        categories_found = {}
        
        for category, keywords in Config.TECHNICAL_KEYWORDS.items():
            found = []
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', resume_lower):
                    found.append(keyword)
            if found:
                categories_found[category] = found
        
        return categories_found


# ===== MODULE 2: METRIC CALCULATOR (PROGRAMMATIC) =====
class MetricCalculator:
    """Calculates objective metrics from resume text"""
    
    def calculate_metrics(self, resume_text: str, experience_years: float) -> Dict[str, int]:
        """Calculate objective scoring metrics"""
        metrics = {
            "clarity": self._calculate_clarity_score(resume_text),
            "impact": self._calculate_impact_score(resume_text),
            "action_verbs": self._calculate_action_verb_score(resume_text),
            "experience_representation": self._calculate_experience_score(resume_text, experience_years)
        }
        return metrics
    
    def _calculate_clarity_score(self, resume_text: str) -> int:
        """Score based on structure and organization"""
        score = 0
        
        # Length check
        word_count = len(resume_text.split())
        if 400 <= word_count <= 800:
            score += 25
        elif 300 <= word_count < 400 or 800 < word_count <= 1000:
            score += 15
        else:
            score += 5
        
        # Section headers check
        sections = ['summary', 'experience', 'education', 'skills', 'technical skills', 
                   'professional experience', 'work experience']
        sections_found = sum(1 for section in sections if section in resume_text.lower())
        score += min(sections_found * 10, 40)
        
        # Formatting consistency (bullet points)
        bullet_patterns = [r'^\s*[•·▪▫◦‣⁃]\s', r'^\s*[-*]\s', r'^\s*\d+\.\s']
        bullet_count = 0
        for line in resume_text.split('\n'):
            if any(re.match(pattern, line) for pattern in bullet_patterns):
                bullet_count += 1
        
        if bullet_count >= 5:
            score += 20
        elif bullet_count >= 3:
            score += 10
        
        # Contact info completeness
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
        has_phone = bool(re.search(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}', resume_text))
        
        if has_email:
            score += 8
        if has_phone:
            score += 7
        
        return min(score, 100)
    
    def _calculate_impact_score(self, resume_text: str) -> int:
        """Score based on quantifiable achievements"""
        score = 0
        
        # Patterns for quantifiable metrics
        metric_patterns = [
            (r'\d+\s*%', 10),  # Percentages - high value
            (r'\$\s*[\d,]+[KMB]?', 10),  # Dollar amounts
            (r'(?:increased|decreased|reduced|improved|saved)\s+(?:\w+\s+)?by\s+\d+', 15),  # Impact statements
            (r'\d+\+?\s*(?:years?|months?|weeks?)', 5),  # Time periods
            (r'(?:managed|led|supervised)\s+(?:team of\s+)?\d+', 8),  # Team size
            (r'\d+x\s+(?:increase|growth|improvement)', 12),  # Multiplier achievements
        ]
        
        for pattern, points in metric_patterns:
            matches = len(re.findall(pattern, resume_text, re.IGNORECASE))
            score += min(matches * points, points * 3)  # Cap at 3 instances per pattern
        
        return min(score, 100)
    
    def _calculate_action_verb_score(self, resume_text: str) -> int:
        """Score based on strong action verb usage"""
        strong_verbs = [
            'achieved', 'implemented', 'developed', 'designed', 'created', 'built',
            'established', 'improved', 'increased', 'decreased', 'reduced', 'optimized',
            'streamlined', 'automated', 'launched', 'spearheaded', 'pioneered',
            'transformed', 'revolutionized', 'architected', 'engineered', 'orchestrated',
            'led', 'managed', 'directed', 'coordinated', 'facilitated', 'mentored'
        ]
        
        # Check for verbs at the beginning of lines/bullets
        score = 0
        lines = resume_text.split('\n')
        
        for line in lines:
            line_lower = line.strip().lower()
            for verb in strong_verbs:
                if line_lower.startswith(verb):
                    score += 5
                    break
        
        return min(score, 100)
    
    def _calculate_experience_score(self, resume_text: str, experience_years: float) -> int:
        """Score based on experience representation matching claimed years"""
        score = 50  # Base score
        
        # Find year patterns (e.g., 2020-2023, Jan 2020, 2020 - Present)
        year_patterns = [
            r'(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}',  # Year ranges
            r'(?:19|20)\d{2}\s*[-–—]\s*[Pp]resent',  # Year to present
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}',  # Month Year
        ]
        
        date_mentions = 0
        for pattern in year_patterns:
            date_mentions += len(re.findall(pattern, resume_text))
        
        # Expect roughly 1 date range per 2-3 years of experience
        expected_positions = max(1, int(experience_years / 2.5))
        
        if date_mentions >= expected_positions:
            score += 30
        elif date_mentions >= expected_positions * 0.7:
            score += 20
        else:
            score += 10
        
        # Check for progression keywords
        progression_keywords = ['promoted', 'advanced', 'progression', 'senior', 'lead', 'principal']
        if experience_years >= 5:
            progression_found = sum(1 for keyword in progression_keywords if keyword in resume_text.lower())
            score += min(progression_found * 5, 20)
        
        return min(score, 100)


# ===== MODULE 3: AI ANALYSIS (FOCUSED QUERIES) =====
class AIAnalyzer:
    """Handles AI-powered analysis with focused, specific queries"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def analyze_professional_summary(self, resume_text: str, experience_years: float) -> Dict[str, Any]:
        """Get AI analysis of professional summary and overall impression"""
        prompt = f"""
        Analyze this resume for a professional with {experience_years} years of experience.
        Focus ONLY on these aspects:
        
        1. First impression (what stands out in 10 seconds)
        2. Professional summary effectiveness
        3. Overall narrative and career story clarity
        4. Whether this person would get an interview (yes/no and why)
        
        Resume:
        {resume_text[:2000]}  # First 2000 chars for context
        
        Return a JSON with keys:
        - "first_impression": string (2-3 sentences)
        - "professional_summary": string (evaluation in 2-3 sentences)  
        - "career_narrative": string (is the career story clear?)
        - "interview_worthy": boolean
        - "interview_reasoning": string (why or why not)
        """
        
        response = self.client.chat.completions.create(
            model=Config.FAST_MODEL,  # Use faster model for this
            response_format={"type": "json_object"},
            temperature=Config.TEMPERATURE,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are an expert recruiter. Provide concise, specific feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_experience_section(self, resume_text: str, experience_years: float) -> Dict[str, Any]:
        """Deep dive into experience section"""
        # Extract just the experience section if possible
        experience_section = self._extract_experience_section(resume_text)
        
        prompt = f"""
        Analyze the work experience section for someone with {experience_years} years of experience.
        
        Evaluate:
        1. Are bullet points achievement-focused or just task lists?
        2. Key strengths demonstrated
        3. Missing elements that should be included
        4. Top 3 bullets that need rewriting (provide specific rewrites)
        
        Experience section:
        {experience_section}
        
        Return JSON with:
        - "achievement_focused": boolean
        - "key_strengths": array of strings (max 5)
        - "missing_elements": array of strings
        - "bullets_to_rewrite": array of objects with "original" and "suggested" keys
        """
        
        response = self.client.chat.completions.create(
            model=Config.MODEL,
            response_format={"type": "json_object"},
            temperature=Config.TEMPERATURE,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "You are an expert resume writer. Be specific and actionable."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
    
    def get_critical_improvements(self, resume_text: str, metrics: Dict[str, int]) -> List[str]:
        """Get top 5 critical improvements based on the full context"""
        prompt = f"""
        Based on this resume and its metrics, provide the TOP 5 most critical improvements.
        
        Current metrics:
        - Clarity: {metrics['clarity']}/100
        - Impact: {metrics['impact']}/100  
        - Action Verbs: {metrics['action_verbs']}/100
        - Experience: {metrics['experience_representation']}/100
        
        Resume excerpt:
        {resume_text[:1500]}
        
        Return JSON with:
        - "critical_improvements": array of 5 specific, actionable improvements
        
        Be direct and specific. Example: "Add quantifiable metrics to all bullet points - currently only 2 out of 15 bullets have numbers"
        """
        
        response = self.client.chat.completions.create(
            model=Config.FAST_MODEL,
            response_format={"type": "json_object"},
            temperature=Config.TEMPERATURE,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a direct, no-nonsense career coach."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("critical_improvements", [])
    
    def _extract_experience_section(self, resume_text: str) -> str:
        """Try to extract just the experience section"""
        lines = resume_text.split('\n')
        experience_start = -1
        experience_end = len(lines)
        
        # Find start of experience section
        for i, line in enumerate(lines):
            if re.search(r'(work\s+experience|professional\s+experience|experience|employment)', 
                        line, re.IGNORECASE):
                experience_start = i
                break
        
        if experience_start == -1:
            return resume_text[:2000]  # Return first part if section not found
        
        # Find end (next major section)
        section_headers = ['education', 'skills', 'certifications', 'awards', 'publications']
        for i in range(experience_start + 1, len(lines)):
            if any(header in lines[i].lower() for header in section_headers):
                experience_end = i
                break
        
        return '\n'.join(lines[experience_start:experience_end])


# ===== MODULE 4: RESULT AGGREGATOR =====
class ResultAggregator:
    """Combines all modular results into final analysis"""
    
    def aggregate_results(self, 
                         keyword_results: List[Dict],
                         metrics: Dict[str, int],
                         summary_analysis: Dict,
                         experience_analysis: Dict,
                         critical_improvements: List[str],
                         experience_years: float) -> Dict[str, Any]:
        """Combine all modular results into final analysis format"""
        
        # Calculate overall score from components
        overall_score = self._calculate_overall_score(metrics, summary_analysis, experience_analysis)
        
        # Build final analysis structure
        final_analysis = {
            "overall_score": overall_score,
            "professional_summary": self._create_professional_summary(
                summary_analysis, experience_years, overall_score
            ),
            "key_metrics": metrics,
            "key_strengths": experience_analysis.get("key_strengths", []),
            "areas_for_improvement": self._format_improvements(experience_analysis),
            "critical_suggestions": critical_improvements,
            "technical_keyword_analysis": keyword_results,
            "interview_decision": {
                "would_interview": summary_analysis.get("interview_worthy", False),
                "reasoning": summary_analysis.get("interview_reasoning", "")
            }
        }
        
        return final_analysis
    
    def _calculate_overall_score(self, metrics: Dict, summary_analysis: Dict, 
                                experience_analysis: Dict) -> int:
        """Calculate overall score from components"""
        # Weighted average of different components
        metric_avg = sum(metrics.values()) / len(metrics)
        
        # Bonus points for interview worthiness
        interview_bonus = 10 if summary_analysis.get("interview_worthy", False) else 0
        
        # Bonus for achievement focus
        achievement_bonus = 10 if experience_analysis.get("achievement_focused", False) else 0
        
        # Calculate final score
        base_score = metric_avg * 0.7  # 70% weight on objective metrics
        subjective_score = (interview_bonus + achievement_bonus) * 0.3  # 30% on subjective
        
        return min(round(base_score + subjective_score), 100)
    
    def _create_professional_summary(self, summary_analysis: Dict, 
                                   experience_years: float, overall_score: int) -> str:
        """Create the professional summary for dashboard"""
        first_impression = summary_analysis.get("first_impression", "")
        interview_worthy = summary_analysis.get("interview_worthy", False)
        
        decision = "would likely proceed to interview" if interview_worthy else "needs improvement before interview consideration"
        
        return f"{first_impression} For a {experience_years}-year professional, this resume scores {overall_score}/100 and {decision}."
    
    def _format_improvements(self, experience_analysis: Dict) -> List[Dict]:
        """Format improvements from experience analysis"""
        improvements = []
        
        # Add bullet rewrite suggestions
        for bullet in experience_analysis.get("bullets_to_rewrite", [])[:3]:
            improvements.append({
                "category": "Bullet Point Improvement",
                "suggestion": "Rewrite this bullet to focus on achievements",
                "example": bullet.get("suggested", "")
            })
        
        # Add missing elements
        for element in experience_analysis.get("missing_elements", [])[:2]:
            improvements.append({
                "category": "Missing Element",
                "suggestion": element,
                "example": "Add this to strengthen your resume"
            })
        
        return improvements


# ===== MAIN ORCHESTRATOR =====
class ResumeAnalyzer:
    """Main class that orchestrates all modules"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.keyword_analyzer = KeywordAnalyzer()
        self.metric_calculator = MetricCalculator()
        self.ai_analyzer = AIAnalyzer(self.client)
        self.aggregator = ResultAggregator()
    
    def analyze_resume(self, resume_text: str, experience_years: float) -> Dict[str, Any]:
        """Run complete modular analysis"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            # Step 1: Extract keywords (fast, no API)
            task = progress.add_task(description="Analyzing technical keywords...", total=None)
            keywords = self.keyword_analyzer.extract_keywords(resume_text)
            
            # Step 2: Calculate metrics (fast, no API)
            progress.update(task, description="Calculating objective metrics...")
            metrics = self.metric_calculator.calculate_metrics(resume_text, experience_years)
            
            # Step 3: AI Analysis - Summary (API call 1)
            progress.update(task, description="Analyzing professional summary...")
            summary_analysis = self.ai_analyzer.analyze_professional_summary(resume_text, experience_years)
            
            # Step 4: AI Analysis - Experience (API call 2)
            progress.update(task, description="Analyzing experience section...")
            experience_analysis = self.ai_analyzer.analyze_experience_section(resume_text, experience_years)
            
            # Step 5: AI Analysis - Critical Improvements (API call 3)
            progress.update(task, description="Identifying critical improvements...")
            critical_improvements = self.ai_analyzer.get_critical_improvements(resume_text, metrics)
            
            # Step 6: Aggregate results
            progress.update(task, description="Compiling final analysis...")
            final_analysis = self.aggregator.aggregate_results(
                keywords, metrics, summary_analysis, experience_analysis,
                critical_improvements, experience_years
            )
        
        return final_analysis


# ===== UTILITY FUNCTIONS =====
def ensure_history_dir_exists():
    """Create analysis history directory if needed"""
    os.makedirs(Config.ANALYSIS_HISTORY_DIR, exist_ok=True)

def save_analysis(analysis_data: Dict, resume_hash: str):
    """Save analysis results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{resume_hash[:10]}_{timestamp}.json"
    filepath = os.path.join(Config.ANALYSIS_HISTORY_DIR, filename)
    
    with open(filepath, "w") as f:
        json.dump(analysis_data, f, indent=4)
    
    console.print(f"\n[bold green]✔ Analysis saved to:[/] [cyan]{filepath}[/]")

def get_resume_hash(resume_text: str) -> str:
    """Generate hash of resume for identification"""
    return hashlib.sha256(resume_text.encode('utf-8')).hexdigest()

def display_dashboard(analysis_data: Dict):
    """Display the analysis dashboard"""
    console.print(Panel(Text("AI Resume Analysis Dashboard (Modular Edition)", 
                            justify="center", style="bold blue")))
    
    # Left Column
    score = analysis_data.get("overall_score", 0)
    score_color = "green" if score >= Config.EXCELLENT_THRESHOLD else "yellow" if score >= Config.GOOD_THRESHOLD else "red"
    score_text = Text(f"{score}/100", style=f"bold {score_color}", justify="center")
    score_panel = Panel(score_text, title="[bold]Overall Score[/bold]", border_style="dim", padding=(1, 2))
    
    # Metrics table
    metrics_table = Table(title="[bold]Key Metrics[/bold]", header_style="bold magenta")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Score", justify="right")
    for metric, value in analysis_data.get("key_metrics", {}).items():
        metric_color = "green" if value >= Config.EXCELLENT_THRESHOLD else "yellow" if value >= Config.GOOD_THRESHOLD else "red"
        display_metric = metric.replace("_", " ").title()
        metrics_table.add_row(display_metric, f"[{metric_color}]{value}[/]")
    
    # Keyword table
    keyword_table = Table(title="[bold]Technical Keywords (Actual Counts)[/bold]", header_style="bold magenta")
    keyword_table.add_column("Technical Term", style="dim")
    keyword_table.add_column("Count", justify="right")
    for item in analysis_data.get("technical_keyword_analysis", []):
        keyword_table.add_row(item.get("term"), str(item.get("count")))
    
    # Interview decision
    interview_data = analysis_data.get("interview_decision", {})
    interview_color = "green" if interview_data.get("would_interview") else "red"
    interview_text = "✓ Interview Candidate" if interview_data.get("would_interview") else "✗ Needs Improvement"
    interview_panel = Panel(
        Text(interview_text, style=f"bold {interview_color}", justify="center"),
        title="[bold]Hiring Decision[/bold]",
        border_style=interview_color
    )
    
    left_column = Group(score_panel, metrics_table, keyword_table, interview_panel)
    
    # Right Column
    summary_panel = Panel(
        Text(analysis_data.get("professional_summary", ""), justify="left"),
        title="[bold]Professional Summary[/bold]"
    )
    
    # Critical suggestions
    critical_text = Text()
    for i, suggestion in enumerate(analysis_data.get("critical_suggestions", [])):
        if i > 0:
            critical_text.append("\n\n")
        critical_text.append(f"💡 {suggestion}")
    critical_panel = Panel(critical_text, title="[bold red]Critical Improvements[/bold red]", border_style="red")
    
    # Strengths
    strengths_text = Text()
    for strength in analysis_data.get("key_strengths", []):
        strengths_text.append(f"✔ {strength}\n")
    strengths_panel = Panel(strengths_text, title="[bold green]Key Strengths[/bold green]", border_style="green")
    
    # Improvements
    improvements_text = Text()
    for i, item in enumerate(analysis_data.get("areas_for_improvement", [])):
        if i > 0:
            improvements_text.append("\n\n")
        improvements_text.append(f"{item.get('category', '')}", style="bold")
        improvements_text.append(f"\n  {item.get('suggestion', '')}")
        if item.get('example'):
            improvements_text.append(f"\n  Example: ", style="dim")
            improvements_text.append(f"{item.get('example', '')}", style="italic")
    improvements_panel = Panel(improvements_text, title="[bold yellow]Specific Improvements[/bold yellow]", border_style="yellow")
    
    right_column = Group(summary_panel, critical_panel, strengths_panel, improvements_panel)
    
    # Layout
    layout_table = Table(show_header=False, show_lines=False, padding=1, expand=True, box=None)
    layout_table.add_column(ratio=1)
    layout_table.add_column(ratio=2)
    layout_table.add_row(left_column, right_column)
    
    console.print(layout_table)


# ===== MAIN FUNCTION =====
def main():
    """Main entry point"""
    load_dotenv(override=True)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]Error: OPENAI_API_KEY not found in .env file[/bold red]")
        return
    
    ensure_history_dir_exists()
    
    # Get user input
    try:
        experience_years = Prompt.ask(
            "[bold yellow]Enter your total years of professional experience[/bold yellow]",
            default="6.5"
        )
        experience_years = float(experience_years)
    except ValueError:
        console.print("[bold red]Invalid input. Please enter a number.[/bold red]")
        return
    
    # Read resume
    try:
        with open(Config.RESUME_FILE_PATH, "r", encoding="utf-8") as f:
            resume_text = f.read()
    except FileNotFoundError:
        console.print(f"[bold red]Error: Resume file not found at '{Config.RESUME_FILE_PATH}'[/bold red]")
        return
    
    # Run analysis
    console.print("\n[bold cyan]Starting modular resume analysis...[/bold cyan]\n")
    
    analyzer = ResumeAnalyzer(api_key)
    analysis_results = analyzer.analyze_resume(resume_text, experience_years)
    
    # Display results
    console.print("\n")
    display_dashboard(analysis_results)
    
    # Save results
    resume_hash = get_resume_hash(resume_text)
    save_analysis(analysis_results, resume_hash)
    
    # Cost estimate
    console.print("\n[dim]Estimated API cost: ~$0.02-0.04 (3 focused API calls)[/dim]")


if __name__ == "__main__":
    main()