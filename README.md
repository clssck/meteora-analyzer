# Meteora Analyzer

A tool to analyze Meteora DLMM (Dynamic Liquidity Market Maker) pools and identify optimal market making opportunities.

## Features

- Real-time data from DexScreener and Meteora APIs
- DLMM-optimized metrics and analysis
- Capital efficiency focus
- Dynamic fee evaluation
- Risk assessment and scoring system
- Investment recommendations
- Optimized batch processing and rate limiting
- Customizable output format (CSV or color-coded Excel)

## Performance Optimizations

### Rate Limiting

- Respects DexScreener's 300 requests/minute limit
- Uses 80% safety buffer (240 effective requests/minute)
- Dynamic delay between requests to prevent rate limit errors

### Batch Processing

- Processes up to 30 addresses per API request
- Runs 20 concurrent requests in parallel
- Progress tracking with ETA and pairs/second metrics
- Memory-efficient response processing

## Scoring System

The tool uses a 100-point scoring system specifically designed for DLMM characteristics:

### Opportunity Score (0-100)

- **Capital Efficiency (25 pts)**

  - Focuses on volume/TVL ratio instead of raw volume
  - Rewards efficient use of liquidity
  - Full points for daily volume/TVL ratio ≥ 0.5
  - Scaled linearly for lower ratios

- **Fee Yield (30 pts)**

  - Emphasizes fee generation relative to TVL
  - Aligns with DLMM's dynamic fee feature
  - Full points for daily fee/TVL ratio ≥ 0.3%
  - Scaled linearly for lower ratios

- **Volatility (15 pts)**

  - Treats volatility as positive (DLMMs increase fees during volatility)
  - 5 pts each for momentum > 1.2 in:
    - 5m timeframe
    - 1h timeframe
    - 6h timeframe

- **Bin Quality (10 pts)**

  - DLMM-specific metric
  - 10 pts for ideal bin step range (20-100)
  - 5 pts for acceptable range (10-200)
  - 0 pts for suboptimal configuration

- **Base Liquidity (10 pts)**

  - Minimum TVL requirement
  - Full points for TVL ≥ $10,000
  - Scaled linearly for lower amounts

- **Safety (10 pts)**
  - 5 pts for being on Jupiter strict list
  - 5 pts for being a bluechip pair

### Risk Rating (1-5, lower is better)

Risk factors specific to DLMM operations:

- Low capital efficiency (volume/TVL < 0.5)
- Poor fee generation (fee/TVL < 0.3%)
- Extreme volatility (momentum > 3x)
- Poor bin configuration (outside 10-300 range)
- Not on Jupiter strict list
- Not a bluechip pair

### Investment Recommendations

- **Strong Buy**: Score ≥ 75 and Risk ≤ 2
  - "Well positioned DLMM"
- **Buy**: Score ≥ 55 and Risk ≤ 3
  - "Good DLMM metrics"
- **Hold**: Score ≥ 35 and Risk ≤ 4
  - "Monitor bin efficiency"
- **Watch**: Everything else
  - "Suboptimal DLMM setup"

## Metrics Explained

### Capital Efficiency Metrics

- Volume/TVL ratios for all timeframes
- Measures how efficiently liquidity is being used
- Higher ratios indicate better capital utilization

### Fee Metrics

- Raw fees for each timeframe
- Fee/TVL ratios to measure yield
- Benefits from DLMM's dynamic fee mechanism
- Higher during periods of volatility

### Bin Configuration

- Optimal bin step range: 20-100
- Acceptable range: 10-200
- Maximum allowed: 300
- Affects concentrated liquidity efficiency

### Momentum Indicators

- 5m/1h ratio: Immediate trend
- 1h/6h ratio: Short-term trend
- 6h/24h ratio: Medium-term trend
- Values > 1.2 indicate good volatility for DLMM
- Values > 3.0 indicate excessive volatility

## Output Formats

### CSV Output

Simple comma-separated values file with all metrics and hyperlinks, suitable for further analysis or importing into other tools.

### Excel Output (XLSX)

Color-coded spreadsheet matching DexScreener's style:

- Volume columns (5m, 1h, 24h): Light blue (#E3F2FD)
- Fee columns (5m, 1h, 24h): Light purple (#F3E5F5)
- Momentum indicators:
  - Green (#E8F5E9) for values > 1 (positive momentum)
  - Red (#FFEBEE) for values < 1 (negative momentum)
- Conditional formatting for scores and risk ratings
- Auto-adjusted column widths (max 50 characters)
- Clickable hyperlinks to:
  - DexScreener price charts
  - Meteora trading interface
  - RugCheck token analysis (for non-bluechip tokens)
- Frozen header row for easy scrolling

## Usage

```bash
# Generate CSV output (default)
python meteora_analyzer.py

# Generate color-coded Excel output
python meteora_analyzer.py --format xlsx
```

The script will:

1. Fetch current market data
2. Filter for active DLMM pools
3. Process data in optimized batches
4. Analyze capital efficiency and metrics
5. Generate scored and sorted recommendations
6. Save results to either:
   - market_opportunities.csv (CSV format)
   - market_opportunities.xlsx (Excel format)

Progress updates are displayed during execution, showing:

- Processed pairs / Total pairs
- Processing speed (pairs/second)
- Estimated time remaining
- Current batch progress
