"""Plain-language help for the Guide page. No claim of profitability anywhere: it explains what each screen means."""
from __future__ import annotations

STAGES = [
    ("Data", "Is the data usable and how current is it? Source, last bar, quality, and whether it is LIVE, DELAYED or HISTORICAL."),
    ("Market", "What kind of market is this? Regime, direction, volatility and key levels. Context only; it does not pick trades."),
    ("Evidence", "Which strategies have a historical record that clears the bars, and why the others do not. Every figure is a past statistic."),
    ("Setup", "Has the selected strategy's entry condition actually occurred on the latest closed bar? A selection is not a setup."),
    ("Risk engine", "An independent check that can veto the trade whatever the ranking says. It names the first check that fails."),
    ("Paper order", "A simulated order, placed only if the risk engine approves. Nothing is sent to a broker in paper mode."),
    ("Outcome", "What happened to earlier paper trades of this strategy, traced from the signal to the exit."),
]

GLOSSARY = {
    "R (risk multiple)": "Profit or loss divided by the amount risked at the stop. +1 R means the trade made what it risked; -1 R means the stop was hit.",
    "Net R": "R after brokerage, taxes and modelled slippage. Gross R ignores these costs and overstates what a trader keeps.",
    "Expectancy": "The average R per trade over many trades. A positive number in the past says nothing certain about the next trade.",
    "Standard error": "How uncertain an average is. Roughly, the true average is probably within two standard errors of the estimate.",
    "Conditional expectancy": "The average net R of a strategy in past situations similar to the current one, shrunk toward the strategy's overall average.",
    "Historical analogue": "A past moment whose measured features resemble now. Analogues describe what happened then; they do not predict what happens next.",
    "Shrinkage": "Pulling a noisy estimate toward a more stable average so a few lucky trades do not look like an edge.",
    "Eligible": "A strategy whose conditional edge, confidence and stability clear fixed thresholds. Eligible does not mean profitable.",
    "NO TRADE": "A valid outcome. It means no strategy qualifies, the data is not good enough, or the top strategies cannot be told apart.",
    "Tie-break": "The top strategies are statistically indistinguishable. A trade taken this way is tagged and sized down; it has no proven edge.",
    "Regime": "A heuristic label for market conditions (trend, range, volatility). Scores are not probabilities and are display-only.",
    "Holdout": "The most recent part of a strategy's trades, set aside and compared with the earlier part to see if the result held up.",
    "Profit factor": "Total gains divided by total losses. Above 1 means gains exceeded losses in that sample; small samples mislead.",
    "Drawdown": "The fall from a previous peak of cumulative results. It shows how painful the worst stretch was.",
    "Multiple testing": "Compare many strategies and the best one will look good partly by luck. The note on the Decision Center says how large an edge must look to stand out.",
    "Parameter sensitivity": "Re-running a strategy with its settings changed by 20%. A result that flips sign was probably fitted to the data.",
    "LIVE / DELAYED / HISTORICAL": "Whether the market is open and the latest bar current (LIVE), open but old (DELAYED), or closed so you see the last session (HISTORICAL).",
    "SIMULATED": "Produced by a simulation (paper fills, modelled charges), not observed in the market.",
    "Risk preview": "The real risk checks run on a candidate without recording anything or issuing an approval.",
    "Veto": "The risk engine refusing a trade. It always names the reason, and the ranking cannot override it.",
}

HOW_TO_READ = [
    "Start at Candidate: the banner says CANDIDATE READY, WAITING FOR SETUP or NO TRADE, and names the stage that stopped it.",
    "Check the data strip first. HISTORICAL means the market is closed and you are looking at the last session.",
    "Open the Decision Center to see why each strategy is ranked, eligible or rejected, with sample sizes and uncertainty.",
    "A historical setup is a past occurrence, a current candidate is a setup on the latest bar, a paper trade is a simulated order.",
    "After trading on paper, use Positions & Orders, then Trace, to compare what was expected with what happened.",
    "Treat every number as a description of the past. Small samples, a few weeks of data and many strategies tried all mean "
    "that an apparent edge may be luck.",
]

LIMITS = [
    "Backtests assume entry at the signal bar's close and model option prices from the underlying; real fills differ.",
    "Only weeks of 5-minute history are cached, which is too little to establish that any selector or strategy has an edge.",
    "Paper trading never touches a broker. Live execution is disabled unless explicitly authorised outside this app's default.",
    "This is research software, not investment advice.",
]
