"""
Enhanced statistical analysis module for TCG-Bench.
Implements bootstrap confidence intervals, effect sizes, power analysis,
and multiple comparison corrections as required by top-tier conferences.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings
from collections import defaultdict

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for TCG-Bench results."""
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 10000):
        """
        Initialize the statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            n_bootstrap: Number of bootstrap samples (default 10000)
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                     statistic_func=np.mean) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for any statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to compute statistic (default: mean)
            
        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        if len(data) == 0:
            return 0.0, 0.0, 0.0
            
        # Original statistic
        point_estimate = statistic_func(data)
        
        # Bootstrap samples
        bootstrap_statistics = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_func(sample))
        
        # Confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
        upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
        
        return point_estimate, lower_bound, upper_bound
    
    def welch_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """
        Perform Welch's t-test for unequal variances.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Dictionary with t-statistic, p-value, and degrees of freedom
        """
        result = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Calculate degrees of freedom for Welch's t-test
        n1, n2 = len(group1), len(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        return {
            't_statistic': result.statistic,
            'p_value': result.pvalue,
            'degrees_of_freedom': df,
            'significant': result.pvalue < self.alpha
        }
    
    def cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        if pooled_std == 0:
            return 0.0
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return d
    
    def interpret_cohen_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            List of corrected p-values
        """
        n_comparisons = len(p_values)
        corrected_p_values = [min(p * n_comparisons, 1.0) for p in p_values]
        return corrected_p_values
    
    def fdr_correction(self, p_values: List[float]) -> Tuple[List[bool], List[float]]:
        """
        Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg method.
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            (reject_null, adjusted_p_values)
        """
        from statsmodels.stats.multitest import multipletests
        
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=self.alpha, 
                                                 method='fdr_bh')
        return reject.tolist(), p_adjusted.tolist()
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                       alpha: Optional[float] = None) -> float:
        """
        Calculate statistical power for given effect size and sample size.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level (default: self.alpha)
            
        Returns:
            Statistical power (0 to 1)
        """
        from statsmodels.stats.power import ttest_power
        
        if alpha is None:
            alpha = self.alpha
            
        power = ttest_power(effect_size, sample_size, alpha, 
                           alternative='two-sided')
        return power
    
    def required_sample_size(self, effect_size: float, desired_power: float = 0.8,
                            alpha: Optional[float] = None) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            desired_power: Desired statistical power (default 0.8)
            alpha: Significance level (default: self.alpha)
            
        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import TTestPower
        
        if alpha is None:
            alpha = self.alpha
            
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size, 
            alpha=alpha, 
            power=desired_power,
            alternative='two-sided'
        )
        return int(np.ceil(sample_size))
    
    def paired_difference_analysis(self, model1_scores: np.ndarray, 
                                  model2_scores: np.ndarray) -> Dict:
        """
        Perform paired difference analysis for model comparisons.
        Accounts for question/game variance by using paired tests.
        
        Args:
            model1_scores: Scores for model 1 (paired with model2)
            model2_scores: Scores for model 2 (paired with model1)
            
        Returns:
            Dictionary with paired test results
        """
        if len(model1_scores) != len(model2_scores):
            raise ValueError("Paired samples must have same length")
        
        differences = model1_scores - model2_scores
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Bootstrap CI for mean difference
        mean_diff, lower_ci, upper_ci = self.bootstrap_confidence_interval(differences)
        
        # Effect size for paired samples (Cohen's d for repeated measures)
        std_diff = np.std(differences, ddof=1)
        cohen_d_paired = mean_diff / std_diff if std_diff > 0 else 0.0
        
        return {
            'mean_difference': mean_diff,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d_paired,
            'effect_size_interpretation': self.interpret_cohen_d(cohen_d_paired),
            'significant': p_value < self.alpha
        }
    
    def irt_analysis(self, response_matrix: np.ndarray) -> Dict:
        """
        Item Response Theory (IRT) analysis for difficulty validation.
        
        Args:
            response_matrix: Matrix of responses (models x games, 1=win, 0=loss)
            
        Returns:
            Dictionary with IRT parameters
        """
        try:
            # Use simple Rasch model approximation
            n_models, n_games = response_matrix.shape
            
            # Model abilities (row means)
            model_abilities = np.mean(response_matrix, axis=1)
            
            # Game difficulties (1 - column means)
            game_difficulties = 1 - np.mean(response_matrix, axis=0)
            
            # Calculate discrimination (how well games separate models)
            discrimination = np.std(response_matrix, axis=0)
            
            # Correlation between predicted and actual
            predicted_probs = []
            actual_outcomes = []
            
            for i in range(n_models):
                for j in range(n_games):
                    # Simple logistic model: P(win) = 1 / (1 + exp(-(ability - difficulty)))
                    prob = 1 / (1 + np.exp(-(model_abilities[i] - game_difficulties[j])))
                    predicted_probs.append(prob)
                    actual_outcomes.append(response_matrix[i, j])
            
            correlation = np.corrcoef(predicted_probs, actual_outcomes)[0, 1]
            
            return {
                'model_abilities': model_abilities.tolist(),
                'game_difficulties': game_difficulties.tolist(),
                'discrimination_indices': discrimination.tolist(),
                'model_fit_correlation': correlation,
                'mean_difficulty': float(np.mean(game_difficulties)),
                'difficulty_variance': float(np.var(game_difficulties))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def variance_analysis(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze variance across different game segments or conditions.
        
        Args:
            data: Dictionary mapping condition names to data arrays
            
        Returns:
            Dictionary with variance analysis results
        """
        results = {}
        
        for condition, values in data.items():
            mean_val, lower_ci, upper_ci = self.bootstrap_confidence_interval(values)
            
            results[condition] = {
                'mean': mean_val,
                'variance': float(np.var(values, ddof=1)),
                'std_dev': float(np.std(values, ddof=1)),
                'ci_lower': lower_ci,
                'ci_upper': upper_ci,
                'n': len(values)
            }
        
        # One-way ANOVA if more than 2 groups
        if len(data) > 2:
            f_stat, p_value = stats.f_oneway(*data.values())
            results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        
        return results
    
    def format_significance_stars(self, p_value: float) -> str:
        """
        Format p-value as significance stars for tables.
        
        Args:
            p_value: P-value to format
            
        Returns:
            Significance stars string
        """
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""
    
    def create_comparison_table(self, models_data: Dict[str, Dict]) -> str:
        """
        Create a formatted comparison table with significance markers.
        
        Args:
            models_data: Dictionary mapping model names to performance data
            
        Returns:
            Formatted table string
        """
        table_lines = []
        table_lines.append("Model | Win Rate | 95% CI | Cohen's d | Significance")
        table_lines.append("-" * 60)
        
        # Get reference model (first in dict)
        ref_model = list(models_data.keys())[0]
        ref_data = models_data[ref_model]
        
        for model, data in models_data.items():
            win_rate = data.get('win_rate', 0)
            ci_lower = data.get('ci_lower', 0)
            ci_upper = data.get('ci_upper', 0)
            
            if model == ref_model:
                cohen_d = "-"
                significance = "ref"
            else:
                # Compare to reference
                d = self.cohen_d(
                    np.array(data.get('raw_scores', [])),
                    np.array(ref_data.get('raw_scores', []))
                )
                cohen_d = f"{d:.2f}"
                
                # Significance test
                test_result = self.welch_t_test(
                    np.array(data.get('raw_scores', [])),
                    np.array(ref_data.get('raw_scores', []))
                )
                significance = self.format_significance_stars(test_result['p_value'])
            
            line = f"{model:15} | {win_rate:6.1f}% | [{ci_lower:.1f}, {ci_upper:.1f}] | {cohen_d:7} | {significance:3}"
            table_lines.append(line)
        
        table_lines.append("-" * 60)
        table_lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        
        return "\n".join(table_lines)


# Utility functions for common TCG-Bench analyses

def analyze_tcg_results(game_results: List[Dict], 
                        confidence_level: float = 0.95) -> Dict:
    """
    Comprehensive analysis of TCG-Bench game results.
    
    Args:
        game_results: List of game result dictionaries
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with comprehensive statistical analysis
    """
    analyzer = StatisticalAnalyzer(confidence_level=confidence_level)
    
    # Group results by model and difficulty
    model_results = defaultdict(lambda: defaultdict(list))
    
    for game in game_results:
        model = game.get('model')
        difficulty = game.get('rollout_count')
        outcome = 1 if game.get('winner') == 'player1' else 0
        model_results[model][difficulty].append(outcome)
    
    # Analyze each model at each difficulty
    analysis = {}
    
    for model in model_results:
        analysis[model] = {}
        
        for difficulty in model_results[model]:
            outcomes = np.array(model_results[model][difficulty])
            
            # Bootstrap confidence interval
            win_rate, ci_lower, ci_upper = analyzer.bootstrap_confidence_interval(outcomes)
            
            analysis[model][difficulty] = {
                'win_rate': win_rate * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'n_games': len(outcomes),
                'raw_scores': outcomes.tolist()
            }
    
    # Model comparisons at each difficulty level
    comparisons = {}
    models = list(model_results.keys())
    
    for difficulty in set(d for m in model_results.values() for d in m):
        comparisons[f'difficulty_{difficulty}'] = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if difficulty in model_results[model1] and difficulty in model_results[model2]:
                    scores1 = np.array(model_results[model1][difficulty])
                    scores2 = np.array(model_results[model2][difficulty])
                    
                    # Welch's t-test
                    test_result = analyzer.welch_t_test(scores1, scores2)
                    
                    # Effect size
                    cohen_d = analyzer.cohen_d(scores1, scores2)
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    comparisons[f'difficulty_{difficulty}'][comparison_key] = {
                        **test_result,
                        'cohen_d': cohen_d,
                        'effect_size': analyzer.interpret_cohen_d(cohen_d)
                    }
    
    return {
        'model_performance': analysis,
        'comparisons': comparisons,
        'summary_statistics': {
            'n_models': len(models),
            'difficulties_tested': sorted(set(d for m in model_results.values() for d in m)),
            'total_games': sum(len(outcomes) for m in model_results.values() 
                             for outcomes in m.values())
        }
    }


def validate_sample_size(effect_size: float = 0.5, desired_power: float = 0.8,
                         current_sample_size: int = 600) -> Dict:
    """
    Validate if current sample size is adequate for TCG-Bench.
    
    Args:
        effect_size: Expected effect size (default: medium)
        desired_power: Desired statistical power
        current_sample_size: Current number of games per condition
        
    Returns:
        Dictionary with validation results
    """
    analyzer = StatisticalAnalyzer()
    
    # Calculate current power
    current_power = analyzer.power_analysis(effect_size, current_sample_size)
    
    # Calculate required sample size
    required_size = analyzer.required_sample_size(effect_size, desired_power)
    
    # Is current size adequate?
    is_adequate = current_sample_size >= required_size
    
    return {
        'current_sample_size': current_sample_size,
        'current_power': current_power,
        'desired_power': desired_power,
        'required_sample_size': required_size,
        'is_adequate': is_adequate,
        'recommendation': f"Current sample size is {'adequate' if is_adequate else 'inadequate'}. "
                         f"{'Consider increasing to ' + str(required_size) + ' games per condition.' if not is_adequate else ''}"
    }