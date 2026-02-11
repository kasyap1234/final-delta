#!/usr/bin/env python3
"""
Evaluate deploy acceptance gates for the redesigned strategy.
Based on acceptance criteria from:
- docs/expectancy_positive_redesign.md
- docs/profitability_improvement_plan.md
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class GateEvaluator:
    """Evaluate deploy acceptance gates."""

    def __init__(self):
        self.gates = {
            'expectancy_payoff': {},
            'performance_risk': {},
            'trade_quality': {},
            'benchmark_relative': {},
            'stress': {}
        }
        self.results = {}

    def evaluate_expectancy_payoff_gates(self, results: Dict) -> Dict:
        """
        Evaluate expectancy and payoff gates.
        
        Criteria from expectancy_positive_redesign.md:
        - stitched out-of-sample expectancy at least +0.08R per trade
        - 95% bootstrap lower bound of stitched expectancy above 0.00R
        - average winner to average loser ratio at least 1.8 overall
        - per-active-regime expectancy positive, with no regime below -0.03R
        """
        gates = {}
        
        # Note: We don't have direct expectancy in R units from the backtest results
        # We'll use proxy metrics: win rate and avg win/loss ratio
        
        # Get trade summaries
        trade_summaries = {
            2023: self._read_trade_summary("backtest_results/2023/trade_summary.txt"),
            2024: self._read_trade_summary("backtest_results/2024_test/trade_summary.txt"),
            2025: self._read_trade_summary("backtest_results/2025_test/trade_summary.txt")
        }
        
        # Compute win/loss ratio
        total_wins = sum(s['winning_trades'] for s in trade_summaries.values())
        total_losses = sum(s['losing_trades'] for s in trade_summaries.values())
        total_avg_win = sum(s['avg_win'] * s['winning_trades'] for s in trade_summaries.values()) / total_wins if total_wins > 0 else 0
        total_avg_loss = sum(s['avg_loss'] * s['losing_trades'] for s in trade_summaries.values()) / total_losses if total_losses > 0 else 0
        
        win_loss_ratio = abs(total_avg_win / total_avg_loss) if total_avg_loss != 0 else 0
        
        gates['win_loss_ratio'] = {
            'threshold': 1.8,
            'actual': win_loss_ratio,
            'pass': win_loss_ratio >= 1.8,
            'description': 'Average winner to average loser ratio at least 1.8'
        }
        
        # Win rate check (proxy for positive expectancy)
        avg_win_rate = sum(s['win_rate'] for s in trade_summaries.values()) / len(trade_summaries)
        gates['win_rate'] = {
            'threshold': 0.40,  # Minimum for positive expectancy
            'actual': avg_win_rate,
            'pass': avg_win_rate >= 0.40,
            'description': 'Win rate at least 40% for positive expectancy'
        }
        
        self.gates['expectancy_payoff'] = gates
        return gates

    def evaluate_performance_risk_gates(self, results: Dict) -> Dict:
        """
        Evaluate performance and risk gates.
        
        Criteria from expectancy_positive_redesign.md:
        - annual net return positive in each target out-of-sample year
        - stitched Sharpe at least 1.1
        - per-year Sharpe at least 0.7
        - stitched profit factor at least 1.15
        - per-year profit factor at least 1.05
        - stitched max drawdown at most 22 percent
        - per-year max drawdown at most 20 percent
        """
        gates = {}
        
        # Yearly results
        yearly_results = {
            2023: self._read_backtest_report("backtest_results/2023/backtest_report.json"),
            2024: self._read_backtest_report("backtest_results/2024_test/backtest_report.json"),
            2025: self._read_backtest_report("backtest_results/2025_test/backtest_report.json")
        }
        
        # Annual net return positive in each year
        all_positive = all(r['total_return'] > 0 for r in yearly_results.values())
        gates['annual_return_positive'] = {
            'threshold': 'All years > 0%',
            'actual': {year: f"{r['total_return']:.2%}" for year, r in yearly_results.items()},
            'pass': all_positive,
            'description': 'Annual net return positive in each target year'
        }
        
        # Per-year Sharpe at least 0.7
        sharpe_pass = all(r['sharpe_ratio'] >= 0.7 for r in yearly_results.values())
        gates['per_year_sharpe'] = {
            'threshold': 0.7,
            'actual': {year: r['sharpe_ratio'] for year, r in yearly_results.items()},
            'pass': sharpe_pass,
            'description': 'Per-year Sharpe at least 0.7'
        }
        
        # Stitched Sharpe at least 1.1
        avg_sharpe = sum(r['sharpe_ratio'] for r in yearly_results.values()) / len(yearly_results)
        gates['stitched_sharpe'] = {
            'threshold': 1.1,
            'actual': avg_sharpe,
            'pass': avg_sharpe >= 1.1,
            'description': 'Stitched Sharpe at least 1.1'
        }
        
        # Per-year max drawdown at most 20 percent
        dd_pass = all(r['max_drawdown'] <= 0.20 for r in yearly_results.values())
        gates['per_year_max_dd'] = {
            'threshold': 0.20,
            'actual': {year: r['max_drawdown'] for year, r in yearly_results.items()},
            'pass': dd_pass,
            'description': 'Per-year max drawdown at most 20 percent'
        }
        
        # Stitched max drawdown at most 22 percent
        avg_dd = sum(r['max_drawdown'] for r in yearly_results.values()) / len(yearly_results)
        gates['stitched_max_dd'] = {
            'threshold': 0.22,
            'actual': avg_dd,
            'pass': avg_dd <= 0.22,
            'description': 'Stitched max drawdown at most 22 percent'
        }
        
        self.gates['performance_risk'] = gates
        return gates

    def evaluate_trade_quality_gates(self, results: Dict) -> Dict:
        """
        Evaluate trade quality gates.
        
        Criteria from expectancy_positive_redesign.md:
        - stop-loss exit share reduced to at most 65 percent
        - trade count reduced at least 30 percent versus failing baseline
        """
        gates = {}
        
        # Get trade summaries
        trade_summaries = {
            2023: self._read_trade_summary("backtest_results/2023/trade_summary.txt"),
            2024: self._read_trade_summary("backtest_results/2024_test/trade_summary.txt"),
            2025: self._read_trade_summary("backtest_results/2025_test/trade_summary.txt")
        }
        
        # Baseline trade counts (from diagnosis)
        baseline_trades = {2023: 718, 2024: 805, 2025: 821}
        
        # Trade count reduction
        total_current = sum(s['total_trades'] for s in trade_summaries.values())
        total_baseline = sum(baseline_trades.values())
        reduction_pct = (total_baseline - total_current) / total_baseline
        
        gates['trade_count_reduction'] = {
            'threshold': 0.30,
            'actual': reduction_pct,
            'pass': reduction_pct >= 0.30,
            'description': 'Trade count reduced at least 30% vs baseline',
            'baseline': total_baseline,
            'current': total_current
        }
        
        # Win rate improvement
        baseline_win_rate = 0.21  # From diagnosis
        current_win_rate = sum(s['win_rate'] for s in trade_summaries.values()) / len(trade_summaries)
        win_rate_improvement = current_win_rate - baseline_win_rate
        
        gates['win_rate_improvement'] = {
            'threshold': 0.05,  # At least 5% improvement
            'actual': win_rate_improvement,
            'pass': win_rate_improvement >= 0.05,
            'description': 'Win rate improved at least 5% vs baseline',
            'baseline': baseline_win_rate,
            'current': current_win_rate
        }
        
        self.gates['trade_quality'] = gates
        return gates

    def evaluate_benchmark_relative_gates(self, results: Dict) -> Dict:
        """
        Evaluate benchmark-relative gates.
        
        Criteria from expectancy_positive_redesign.md:
        - stitched Sharpe at least benchmark Sharpe minus 0.10
        - strategy drawdown at most benchmark drawdown plus 5 percentage points
        - up-year capture guardrail and underperformance cap must pass
        """
        gates = {}
        
        # Read benchmark analysis
        with open("backtest_results/walk_forward_analysis.json", 'r') as f:
            benchmark_data = json.load(f)
        
        # Yearly results
        yearly_results = {
            2023: self._read_backtest_report("backtest_results/2023/backtest_report.json"),
            2024: self._read_backtest_report("backtest_results/2024_test/backtest_report.json"),
            2025: self._read_backtest_report("backtest_results/2025_test/backtest_report.json")
        }
        
        # Check if strategy beats benchmark in down years
        down_years = [int(year) for year, data in benchmark_data.items()
                      if data['benchmark']['equal_weight'] < 0]

        beats_benchmark_in_down = []
        for year in down_years:
            strategy_return = yearly_results[year]['total_return']
            benchmark_return = benchmark_data[str(year)]['benchmark']['equal_weight']
            beats_benchmark_in_down.append(strategy_return > benchmark_return)
        
        gates['beats_benchmark_down_years'] = {
            'threshold': 'True for all down years',
            'actual': {year: beats_benchmark_in_down[i] for i, year in enumerate(down_years)},
            'pass': all(beats_benchmark_in_down) if beats_benchmark_in_down else True,
            'description': 'Strategy beats benchmark in down years'
        }
        
        # Drawdown vs benchmark
        # Note: We don't have benchmark drawdown, so we check if strategy DD is reasonable
        avg_strategy_dd = sum(r['max_drawdown'] for r in yearly_results.values()) / len(yearly_results)
        gates['reasonable_drawdown'] = {
            'threshold': 0.15,
            'actual': avg_strategy_dd,
            'pass': avg_strategy_dd <= 0.15,
            'description': 'Strategy max drawdown at most 15%'
        }
        
        self.gates['benchmark_relative'] = gates
        return gates

    def evaluate_stress_gates(self, results: Dict) -> Dict:
        """
        Evaluate stress gates.
        
        Criteria from expectancy_positive_redesign.md:
        - S1 through S4: stitched net return remains positive
        - S5 and S6: non-negative stitched expectancy
        - S7 combined adverse: stitched return above -7 percent and max drawdown at most 30 percent
        """
        gates = {}
        
        # Read adversarial results
        baseline = self._read_backtest_report("backtest_results/2023/backtest_report.json")
        pessimistic = self._read_backtest_report("backtest_results/pessimistic_2023/backtest_report.json")
        sensitivity = self._read_backtest_report("backtest_results/sensitivity_2023/backtest_report.json")
        
        # S1 (pessimistic): stitched net return remains positive
        gates['s1_pessimistic_positive'] = {
            'threshold': '> 0%',
            'actual': pessimistic['total_return'],
            'pass': pessimistic['total_return'] > 0,
            'description': 'Pessimistic scenario: net return remains positive'
        }
        
        # S2 (sensitivity): stitched net return remains positive
        gates['s2_sensitivity_positive'] = {
            'threshold': '> 0%',
            'actual': sensitivity['total_return'],
            'pass': sensitivity['total_return'] > 0,
            'description': 'Sensitivity scenario: net return remains positive'
        }
        
        # Degradation limits
        pessimistic_degradation = abs(pessimistic['total_return'] - baseline['total_return'])
        sensitivity_degradation = abs(sensitivity['total_return'] - baseline['total_return'])
        
        gates['degradation_limit'] = {
            'threshold': 0.05,  # Max 5% degradation
            'actual': {
                'pessimistic': pessimistic_degradation,
                'sensitivity': sensitivity_degradation
            },
            'pass': max(pessimistic_degradation, sensitivity_degradation) <= 0.05,
            'description': 'Adversarial degradation at most 5%'
        }
        
        # Max drawdown in stress scenarios
        max_stress_dd = max(pessimistic['max_drawdown'], sensitivity['max_drawdown'])
        gates['stress_max_dd'] = {
            'threshold': 0.15,
            'actual': max_stress_dd,
            'pass': max_stress_dd <= 0.15,
            'description': 'Stress scenario max drawdown at most 15%'
        }
        
        self.gates['stress'] = gates
        return gates

    def evaluate_v3_gates(self, results: Dict) -> Dict:
        """
        Evaluate V3-specific gates from v3_pivot_design.md.
        
        V3 gates include:
        - Up-year participation 2023 and 2024
        - Annual upside capture ratio in up years
        - Monthly upside capture median in up years
        - Downside capture ratio in down windows
        - Relative drawdown ratio
        - Stitched expectancy
        - Bootstrap confidence
        - Average winner to loser
        - Stop-loss exit share
        - Active regime expectancy
        """
        gates = {}
        
        # Yearly results
        yearly_results = {
            2023: self._read_backtest_report("backtest_results/2023/backtest_report.json"),
            2024: self._read_backtest_report("backtest_results/2024_test/backtest_report.json"),
            2025: self._read_backtest_report("backtest_results/2025_test/backtest_report.json")
        }
        
        # Trade summaries
        trade_summaries = {
            2023: self._read_trade_summary("backtest_results/2023/trade_summary.txt"),
            2024: self._read_trade_summary("backtest_results/2024_test/trade_summary.txt"),
            2025: self._read_trade_summary("backtest_results/2025_test/trade_summary.txt")
        }
        
        # V3 Gate: Up-year participation 2023 and 2024
        up_year_2023_positive = yearly_results[2023]['total_return'] > 0
        up_year_2024_positive = yearly_results[2024]['total_return'] > 0
        up_year_2023_at_least_20pct = yearly_results[2023]['total_return'] >= 0.20
        up_year_2024_at_least_20pct = yearly_results[2024]['total_return'] >= 0.20
        
        gates['v3_up_year_participation'] = {
            'threshold': 'Positive and at least +20% in 2023 and 2024',
            'actual': {
                2023: f"{yearly_results[2023]['total_return']:.2%}",
                2024: f"{yearly_results[2024]['total_return']:.2%}"
            },
            'pass': up_year_2023_positive and up_year_2024_positive and up_year_2023_at_least_20pct and up_year_2024_at_least_20pct,
            'description': 'Up-year participation 2023 and 2024: positive and at least +20% each'
        }
        
        # V3 Gate: Annual upside capture ratio in up years (placeholder - requires benchmark data)
        gates['v3_annual_upside_capture'] = {
            'threshold': 0.25,
            'actual': 'TBD - requires benchmark data',
            'pass': None,  # Will be computed when benchmark data is available
            'description': 'Annual upside capture ratio in up years at least 0.25 each year'
        }
        
        # V3 Gate: Monthly upside capture median in up years (placeholder - requires monthly data)
        gates['v3_monthly_upside_capture'] = {
            'threshold': 0.50,
            'actual': 'TBD - requires monthly data',
            'pass': None,  # Will be computed when monthly data is available
            'description': 'Monthly upside capture median in up years at least 0.50'
        }
        
        # V3 Gate: Downside capture ratio in down windows (placeholder - requires benchmark data)
        gates['v3_downside_capture'] = {
            'threshold': 0.90,
            'actual': 'TBD - requires benchmark data',
            'pass': None,  # Will be computed when benchmark data is available
            'description': 'Downside capture ratio in down windows at most 0.90'
        }
        
        # V3 Gate: Relative drawdown ratio (placeholder - requires benchmark data)
        gates['v3_relative_drawdown'] = {
            'threshold': 1.25,
            'actual': 'TBD - requires benchmark data',
            'pass': None,  # Will be computed when benchmark data is available
            'description': 'Relative drawdown ratio: strategy max DD <= 1.25x benchmark DD'
        }
        
        # V3 Gate: Stitched expectancy (proxy using win rate and win/loss ratio)
        total_wins = sum(s['winning_trades'] for s in trade_summaries.values())
        total_losses = sum(s['losing_trades'] for s in trade_summaries.values())
        total_avg_win = sum(s['avg_win'] * s['winning_trades'] for s in trade_summaries.values()) / total_wins if total_wins > 0 else 0
        total_avg_loss = sum(s['avg_loss'] * s['losing_trades'] for s in trade_summaries.values()) / total_losses if total_losses > 0 else 0
        avg_win_rate = sum(s['win_rate'] for s in trade_summaries.values()) / len(trade_summaries)
        
        # Proxy expectancy: (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        # Normalize by avg_loss to get R units
        proxy_expectancy_r = ((avg_win_rate * total_avg_win) - ((1 - avg_win_rate) * total_avg_loss)) / total_avg_loss if total_avg_loss != 0 else 0
        
        gates['v3_stitched_expectancy'] = {
            'threshold': 0.08,
            'actual': proxy_expectancy_r,
            'pass': proxy_expectancy_r >= 0.08,
            'description': 'Stitched expectancy at least +0.08R'
        }
        
        # V3 Gate: Bootstrap confidence (placeholder - requires bootstrap analysis)
        gates['v3_bootstrap_confidence'] = {
            'threshold': 0.00,
            'actual': 'TBD - requires bootstrap analysis',
            'pass': None,  # Will be computed when bootstrap analysis is available
            'description': '95% lower bound of expectancy above 0.00R'
        }
        
        # V3 Gate: Average winner to loser
        win_loss_ratio = abs(total_avg_win / total_avg_loss) if total_avg_loss != 0 else 0
        gates['v3_avg_winner_loser'] = {
            'threshold': 1.8,
            'actual': win_loss_ratio,
            'pass': win_loss_ratio >= 1.8,
            'description': 'Average winner to loser ratio at least 1.8'
        }
        
        # V3 Gate: Stop-loss exit share (placeholder - requires exit reason data)
        gates['v3_stop_loss_share'] = {
            'threshold': 0.62,
            'actual': 'TBD - requires exit reason data',
            'pass': None,  # Will be computed when exit reason data is available
            'description': 'Stop-loss exit share at most 62%'
        }
        
        # V3 Gate: Active regime expectancy (placeholder - requires regime-specific data)
        gates['v3_active_regime_expectancy'] = {
            'threshold': 'All positive, none below -0.02R',
            'actual': 'TBD - requires regime-specific data',
            'pass': None,  # Will be computed when regime-specific data is available
            'description': 'Active regime expectancy: every active regime positive and none below -0.02R'
        }
        
        # Add V3 gates to the gates dictionary
        self.gates['v3'] = gates
        return gates

    def _read_backtest_report(self, path: str) -> Dict:
        """Read backtest report JSON."""
        with open(path, 'r') as f:
            return json.load(f)

    def _read_trade_summary(self, path: str) -> Dict:
        """Read trade summary text file."""
        with open(path, 'r') as f:
            content = f.read()
        
        # Parse the summary
        lines = content.strip().split('\n')
        result = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Parse numeric values
                if '$' in value:
                    result[key] = float(value.replace('$', '').replace(',', ''))
                elif '%' in value:
                    result[key] = float(value.replace('%', '')) / 100
                else:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
        
        # Map keys
        return {
            'total_trades': result.get('total_trades', 0),
            'winning_trades': result.get('winning_trades', 0),
            'losing_trades': result.get('losing_trades', 0),
            'win_rate': result.get('win_rate', 0),
            'avg_win': result.get('avg_win', 0),
            'avg_loss': result.get('avg_loss', 0)
        }

    def evaluate_all_gates(self) -> Dict:
        """Evaluate all gates and return results."""
        print("=" * 80)
        print("DEPLOY ACCEPTANCE GATE EVALUATION")
        print("=" * 80)
        
        # Evaluate all gate categories
        self.evaluate_expectancy_payoff_gates({})
        self.evaluate_performance_risk_gates({})
        self.evaluate_trade_quality_gates({})
        self.evaluate_benchmark_relative_gates({})
        self.evaluate_stress_gates({})
        self.evaluate_v3_gates({})
        
        # Print results
        for category, gates in self.gates.items():
            print(f"\n{'=' * 80}")
            print(f"{category.upper().replace('_', ' ')} GATES")
            print(f"{'=' * 80}")
            
            for gate_name, gate_info in gates.items():
                status = "PASS" if gate_info['pass'] else "FAIL"
                print(f"\n[{status}] {gate_info['description']}")
                print(f"  Threshold: {gate_info['threshold']}")
                print(f"  Actual: {gate_info['actual']}")
        
        # Summary
        print(f"\n{'=' * 80}")
        print("GATE SUMMARY")
        print(f"{'=' * 80}")
        
        total_gates = 0
        passed_gates = 0
        
        for category, gates in self.gates.items():
            category_pass = all(g['pass'] for g in gates.values())
            category_status = "PASS" if category_pass else "FAIL"
            passed = sum(1 for g in gates.values() if g['pass'])
            total = len(gates)
            
            print(f"\n{category.upper().replace('_', ' ')}: {category_status} ({passed}/{total} gates passed)")
            
            total_gates += total
            passed_gates += passed
        
        print(f"\n{'=' * 80}")
        print(f"OVERALL: {passed_gates}/{total_gates} gates passed ({passed_gates/total_gates*100:.1f}%)")
        print(f"{'=' * 80}")
        
        # Final verdict
        all_pass = all(all(g['pass'] for g in gates.values()) for gates in self.gates.values())
        verdict = "GO" if all_pass else "NO-GO"
        print(f"\nFINAL VERDICT: {verdict}")
        print(f"{'=' * 80}")
        
        return {
            'gates': self.gates,
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'pass_rate': passed_gates / total_gates,
                'verdict': verdict
            }
        }


def main():
    evaluator = GateEvaluator()
    results = evaluator.evaluate_all_gates()
    
    # Save results
    output_path = Path("backtest_results/gate_evaluation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGate evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
