import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def compute_confidence_interval(data) -> tuple:
    a = np.array(data)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    mean = np.mean(a)
    # with only one sample, CI width is zero
    if n == 1:
        return mean, 0.0
    std_err = np.std(a, ddof=1) / np.sqrt(n)
    ci_half_width = 1.96 * std_err
    return mean, ci_half_width


class MultilingualAnalyzer:
    """Enhanced multilingual analysis with GAP metrics."""
    
    def __init__(self):
        """Initialize the multilingual analyzer."""
        pass
    
    def calculate_gap_metric(self, performance_by_language: Dict[str, float]) -> float:
        """
        Calculate GAP metric for multilingual performance.
        GAP = Σ(max(s(en) - s(l), 0))/(n-1) where n is number of languages.
        
        Args:
            performance_by_language: Dict mapping language code to performance score
            
        Returns:
            GAP metric value
        """
        if "en" not in performance_by_language:
            raise ValueError("English ('en') must be included as reference language")
        
        en_score = performance_by_language["en"]
        gap_sum = 0
        other_langs = 0
        
        for lang, score in performance_by_language.items():
            if lang != "en":
                gap_sum += max(en_score - score, 0)
                other_langs += 1
        
        if other_langs == 0:
            return 0.0
        
        return gap_sum / other_langs
    
    def calculate_relative_gap(self, score_en: float, score_other: float) -> float:
        """
        Calculate relative performance gap between English and another language.
        
        Args:
            score_en: English performance score
            score_other: Other language performance score
            
        Returns:
            Relative gap as percentage
        """
        if score_en == 0:
            return 0.0
        
        return ((score_en - score_other) / score_en) * 100
    
    def calculate_spearman_correlation(self, scores_lang1: List[float], 
                                      scores_lang2: List[float]) -> Tuple[float, float]:
        """
        Calculate Spearman rank correlation between two languages.
        
        Args:
            scores_lang1: Performance scores for language 1
            scores_lang2: Performance scores for language 2
            
        Returns:
            (correlation, p_value)
        """
        if len(scores_lang1) != len(scores_lang2):
            raise ValueError("Score lists must have same length")
        
        if len(scores_lang1) < 3:
            return 0.0, 1.0  # Not enough data
        
        correlation, p_value = stats.spearmanr(scores_lang1, scores_lang2)
        return correlation, p_value
    
    def calculate_retention_rate(self, base_concepts: List[str], 
                                translated_concepts: List[str]) -> float:
        """
        Calculate strategic concept retention rate between languages.
        
        Args:
            base_concepts: List of strategic concepts in base language
            translated_concepts: List of concepts preserved in translation
            
        Returns:
            Retention rate (0 to 1)
        """
        if not base_concepts:
            return 0.0
        
        base_set = set(base_concepts)
        translated_set = set(translated_concepts)
        
        retained = base_set.intersection(translated_set)
        retention_rate = len(retained) / len(base_set)
        
        return retention_rate
    
    def analyze_linguistic_bias(self, results_by_language: Dict[str, List[float]]) -> Dict:
        """
        Comprehensive analysis of linguistic bias in model performance.
        
        Args:
            results_by_language: Dict mapping language to list of performance scores
            
        Returns:
            Dictionary with bias analysis
        """
        # Calculate mean performance per language
        mean_scores = {lang: np.mean(scores) for lang, scores in results_by_language.items()}
        
        # Calculate GAP metric
        gap_metric = self.calculate_gap_metric(mean_scores)
        
        # Calculate pairwise correlations
        correlations = {}
        for lang1 in results_by_language:
            for lang2 in results_by_language:
                if lang1 < lang2:  # Avoid duplicates
                    corr, p_val = self.calculate_spearman_correlation(
                        results_by_language[lang1],
                        results_by_language[lang2]
                    )
                    correlations[f"{lang1}_vs_{lang2}"] = {
                        "correlation": corr,
                        "p_value": p_val,
                        "significant": p_val < 0.05
                    }
        
        # Calculate relative gaps
        relative_gaps = {}
        if "en" in mean_scores:
            en_score = mean_scores["en"]
            for lang, score in mean_scores.items():
                if lang != "en":
                    relative_gaps[lang] = self.calculate_relative_gap(en_score, score)
        
        # Assess bias level
        bias_assessment = self._assess_bias_level(gap_metric, relative_gaps)
        
        return {
            "mean_scores": mean_scores,
            "gap_metric": gap_metric,
            "relative_gaps": relative_gaps,
            "correlations": correlations,
            "bias_assessment": bias_assessment,
            "recommendations": self._generate_bias_recommendations(bias_assessment)
        }
    
    def _assess_bias_level(self, gap_metric: float, relative_gaps: Dict[str, float]) -> str:
        """Assess the level of linguistic bias."""
        max_gap = max(relative_gaps.values()) if relative_gaps else 0
        
        if gap_metric < 5 and max_gap < 10:
            return "Minimal bias"
        elif gap_metric < 10 and max_gap < 20:
            return "Low bias"
        elif gap_metric < 20 and max_gap < 30:
            return "Moderate bias"
        elif gap_metric < 30 and max_gap < 40:
            return "High bias"
        else:
            return "Severe bias"
    
    def _generate_bias_recommendations(self, bias_level: str) -> List[str]:
        """Generate recommendations based on bias level."""
        recommendations = []
        
        if bias_level in ["High bias", "Severe bias"]:
            recommendations.append("Urgent: Address linguistic bias in model training")
            recommendations.append("Consider multilingual fine-tuning")
            recommendations.append("Validate translation quality")
        elif bias_level == "Moderate bias":
            recommendations.append("Monitor linguistic performance gaps")
            recommendations.append("Consider targeted improvements for underperforming languages")
        else:
            recommendations.append("Maintain current multilingual support")
            recommendations.append("Continue monitoring for bias emergence")
        
        return recommendations
    
    def generate_cross_linguistic_report(self, game_results: List[Dict]) -> Dict:
        """
        Generate comprehensive cross-linguistic performance report.
        
        Args:
            game_results: List of game result dictionaries with language info
            
        Returns:
            Comprehensive report dictionary
        """
        # Group results by language
        results_by_lang = {}
        for result in game_results:
            lang = result.get("language", "en")
            if lang not in results_by_lang:
                results_by_lang[lang] = []
            
            # Extract win/loss as binary
            win = 1 if result.get("winner") == "player1" else 0
            results_by_lang[lang].append(win)
        
        # Perform bias analysis
        bias_analysis = self.analyze_linguistic_bias(results_by_lang)
        
        # Calculate detailed metrics per language
        language_metrics = {}
        for lang, scores in results_by_lang.items():
            language_metrics[lang] = {
                "mean_win_rate": np.mean(scores) * 100,
                "std_dev": np.std(scores) * 100,
                "num_games": len(scores),
                "confidence_interval": self._calculate_ci(scores)
            }
        
        return {
            "languages_tested": list(results_by_lang.keys()),
            "language_metrics": language_metrics,
            "bias_analysis": bias_analysis,
            "summary": {
                "gap_metric": bias_analysis["gap_metric"],
                "bias_level": bias_analysis["bias_assessment"],
                "worst_gap": max(bias_analysis["relative_gaps"].values()) if bias_analysis["relative_gaps"] else 0
            }
        }
    
    def _calculate_ci(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate 95% confidence interval."""
        if not scores:
            return (0, 0)
        
        mean = np.mean(scores)
        std_err = np.std(scores, ddof=1) / np.sqrt(len(scores))
        ci = 1.96 * std_err
        
        return ((mean - ci) * 100, (mean + ci) * 100)
