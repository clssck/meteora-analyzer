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

- **Capital Efficiency (45 pts)**

  - Focuses on volume/TVL ratio instead of raw volume
  - Rewards efficient use of liquidity
  - Full points for daily volume/TVL ratio ≥ 5.0
  - Scaled linearly for lower ratios

- **Fee Yield (20 pts)**

  - Emphasizes fee generation relative to TVL
  - Aligns with DLMM's dynamic fee feature
  - Full points for daily fee/TVL ratio ≥ 0.2%
  - Scaled linearly for lower ratios

- **Volatility (10 pts)**

  - Treats moderate volatility as positive (DLMMs increase fees during volatility)
  - Up to 5 pts each for momentum in:
    - 5m timeframe (0.5-2.0 range)
    - 1h timeframe (0.5-2.0 range)
    - 6h timeframe (0.5-2.0 range)
  - Penalties for extreme volatility

- **Base Liquidity (15 pts)**

  - Minimum TVL requirement
  - Full points for TVL ≥ $100,000
  - Scaled linearly for lower amounts
  - Additional penalties for very low TVL

- **Bin Quality (10 pts)**

  - DLMM-specific metric
  - 10 pts for ideal bin step range (20-100)
  - 5 pts for acceptable range (10-200)
  - 0 pts for suboptimal configuration

### Risk Rating (1-5, higher is riskier)

Risk factors specific to DLMM operations:

1 - Very Safe - Bluechip pair with optimal metrics
2 - Safe - Good metrics with manageable volatility
3 - Moderate - Decent opportunity but watch bin positioning
4 - High - Multiple concerns need monitoring
5 - Very High - Multiple efficiency/safety concerns

Risk factors considered:

- Low capital efficiency (volume/TVL < 1.0)
- Poor fee generation (fee/TVL < 0.1%)
- Extreme volatility (momentum > 3x)
- Poor bin configuration (outside 10-200 range)
- Low TVL (< $10,000)
- Not on Jupiter strict list
- Not a bluechip pair

### Investment Recommendations

- **Strong Buy**: Score ≥ 90 and Risk ≤ 2
  - "Well positioned DLMM"
- **Buy**: Score ≥ 80 and Risk ≤ 3
  - "Good DLMM metrics"
- **Hold**: Score ≥ 70 and Risk ≤ 4
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

Comprehensive comma-separated values file with:

- All core metrics and calculations
- Risk factors and scoring details
- Hyperlinks to:
  - RugCheck token analysis
  - DexScreener price charts
  - GeckoTerminal analytics
  - DexTools market data
  - Meteora trading interface

### Excel Output (XLSX)

Color-coded spreadsheet with enhanced formatting:

- Volume columns (5m, 1h, 24h): Light blue (#E3F2FD)
- Fee columns (5m, 1h, 24h): Light purple (#F3E5F5)
- Momentum indicators:
  - Green (#E8F5E9) for optimal range (0.5-2.0)
  - Yellow for borderline values
  - Red (#FFEBEE) for concerning values
- Conditional formatting for scores and risk ratings
- Auto-adjusted column widths
- Frozen header row for easy scrolling
- Integrated market analysis links

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

## Advanced Usage

### Command Line Arguments

```bash
# Analyze specific pools only
python meteora_analyzer.py --pools <pool1,pool2,pool3>

# Set custom TVL threshold
python meteora_analyzer.py --min-tvl 50000

# Adjust risk tolerance
python meteora_analyzer.py --max-risk 3

# Custom output directory
python meteora_analyzer.py --output-dir /path/to/dir

# Debug mode with verbose logging
python meteora_analyzer.py --debug
```

### Environment Variables

```bash
# Rate limiting
export METEORA_MAX_REQUESTS=300
export METEORA_SAFETY_BUFFER=0.8

# API endpoints
export DEXSCREENER_API_URL=custom_url
export METEORA_API_URL=custom_url

# Output formatting
export DEFAULT_OUTPUT_FORMAT=xlsx
```

## Practical Tips

### Best Times to Run Analysis

- Run during UTC 12:00-20:00 for highest liquidity periods
- Avoid running during known high-volatility events
- Consider running multiple times per day to catch market shifts

### Optimal Pool Selection

1. **High-Quality Pairs**

   - Both tokens on Jupiter strict list
   - Combined market cap > $10M
   - Consistent trading history

2. **Ideal Configuration**

   - Bin step: 20-50 for stable pairs
   - Bin step: 50-100 for volatile pairs
   - Active price within 20% of bin range

3. **Red Flags to Avoid**
   - Extreme price impact (>2% for 1000 USD)
   - Single-sided liquidity (>80% imbalance)
   - Inactive bins for >48 hours

### Position Management

- Size positions based on bin concentration
- Monitor bin utilization rates
- Rebalance when price moves >30% from entry
- Consider gas costs in rebalancing decisions

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**

   ```
   Solution: Increase safety buffer or reduce concurrent requests
   export METEORA_SAFETY_BUFFER=0.7
   ```

2. **Stale Data**

   ```
   Solution: Clear cache directory
   rm -rf ~/.meteora/cache/*
   ```

3. **High Memory Usage**

   ```
   Solution: Reduce batch size
   python meteora_analyzer.py --batch-size 15
   ```

4. **Missing Pool Data**
   ```
   Solution: Verify pool address and retry
   python meteora_analyzer.py --retry-failed
   ```

### Performance Optimization

1. **Speed Up Analysis**

   - Use SSD for cache directory
   - Increase concurrent requests if stable
   - Filter out low-TVL pools early

2. **Reduce API Load**

   - Cache responses locally
   - Use compressed data format
   - Implement smart retry logic

3. **Memory Efficiency**
   - Stream large datasets
   - Clean up old cache files
   - Use generator patterns

### Error Codes

- `E001`: Rate limit exceeded
- `E002`: Invalid pool address
- `E003`: API connection failed
- `E004`: Data parsing error
- `E005`: Output write failed

## Data Sources

### Primary APIs

- DexScreener: Market data, volumes
- Meteora: Pool configs, bin data
- Jupiter: Token validation
- Solana RPC: Chain data

### Reliability Metrics

- DexScreener: 99.9% uptime
- Meteora API: 99.8% uptime
- Update frequency: 15s
- Data latency: ~3s

## Support and Resources

### Official Links

- [Meteora Docs](https://docs.meteora.ag/)
- [DexScreener API](https://docs.dexscreener.com/)
- [Jupiter API](https://docs.jup.ag/)

### Community

- [Meteora Discord](https://discord.gg/meteora)
- [Telegram Updates](https://t.me/meteora_updates)
- [Github Issues](https://github.com/meteora-ag/dlmm)

### Updates

- Check releases page for latest versions
- Enable update notifications
- Review changelog before upgrading
