import numpy as np
import pandas as pd
import config
from security import logger, safe_atomic_write, send_push_notification

class PortfolioBacktester:
    """
    Layer 3 Multi-Asset Portfolio Execution Simulator.
    Features:
      - Dynamic volatility position sizing
      - Slippage penalty deductions on all exits
      - Portfolio-wide weekly drawdown circuit breaker (5% emergency halt)
    """
    def __init__(self, initial_capital=config.INITIAL_CAPITAL, 
                 max_alloc=config.MAX_ALLOCATION_PER_TRADE,
                 tp_mult=config.TP_ATR_MULT, 
                 sl_mult_long=config.SL_ATR_MULT_LONG,
                 sl_mult_short=config.SL_ATR_MULT_SHORT,
                 enable_breakeven=config.ENABLE_BREAKEVEN,
                 slippage_penalty=config.SLIPPAGE_PENALTY_PCT,
                 drawdown_limit=config.WEEKLY_DRAWDOWN_LIMIT):
        self.initial_capital = initial_capital
        self.max_alloc = max_alloc
        self.tp_mult = tp_mult
        self.sl_mult_long = sl_mult_long
        self.sl_mult_short = sl_mult_short
        self.enable_breakeven = enable_breakeven
        self.slippage_penalty = slippage_penalty
        self.drawdown_limit = drawdown_limit
        
        self.trade_log = []
        self.portfolio_equity = []
        self.circuit_breaker_tripped = False
        self.circuit_breaker_date = None

    def run(self, test_dfs, test_signals_dict, test_allowance_dict, test_probs_dict=None):
        """
        Runs the portfolio simulation with risk filters, slippage, and circuit breaker.
        """
        self.trade_log = []
        self.portfolio_equity = []
        self.circuit_breaker_tripped = False
        self.circuit_breaker_date = None
        
        all_dates = sorted(list(set().union(*(df.index for df in test_dfs.values()))))
        
        shared_cash = self.initial_capital
        positions = {}  # {ticker: pos_dict}
        
        for date in all_dates:
            # 1. CALCULATE CURRENT DAY'S EQUITY (Mark-to-Market)
            current_equity = shared_cash
            for ticker in positions:
                df = test_dfs[ticker]
                pos = positions[ticker]
                is_long = pos.get('direction', 'long') == 'long'
                if date in df.index:
                    curr_price = df.loc[date, 'Close']
                else:
                    curr_price = pos['entry_price']
                
                if is_long:
                    current_equity += pos['units'] * curr_price
                else:
                    current_equity += (pos['units'] * pos['entry_price']) + (pos['units'] * (pos['entry_price'] - curr_price))
            
            # Record equity
            self.portfolio_equity.append(current_equity)
            
            # 2. EVALUATE WEEKLY DRAWDOWN CIRCUIT BREAKER
            # Check trailing 7-day (7-bar) equity peak
            if len(self.portfolio_equity) >= 2:
                lookback = min(7, len(self.portfolio_equity))
                rolling_peak = max(self.portfolio_equity[-lookback:])
                current_drawdown = (rolling_peak - current_equity) / rolling_peak
                
                if current_drawdown >= self.drawdown_limit and not self.circuit_breaker_tripped:
                    self.circuit_breaker_tripped = True
                    self.circuit_breaker_date = date
                    msg = (
                        f"🚨 **CIRCUIT BREAKER TRIGGERED** on {date.date()}!\n"
                        f"Weekly drawdown of {current_drawdown:.2%} exceeded limit of {self.drawdown_limit:.2%}.\n"
                        f"Emergency exiting all positions and halting all trading."
                    )
                    logger.critical(msg)
                    send_push_notification(msg)
            
            # 3. IF CIRCUIT BREAKER TRIPPED: FORCE EMERGENCY EXITS AND ABORT
            if self.circuit_breaker_tripped:
                for ticker in list(positions.keys()):
                    pos = positions[ticker]
                    df = test_dfs[ticker]
                    close_val = df.loc[date, 'Close'] if date in df.index else pos['entry_price']
                    is_long = pos.get('direction', 'long') == 'long'
                    
                    if is_long:
                        pnl_pct = (close_val - pos['entry_price']) / pos['entry_price'] - self.slippage_penalty
                        exit_val = pos['units'] * close_val * (1.0 - self.slippage_penalty)
                        pnl_usd = exit_val - (pos['units'] * pos['entry_price'])
                        shared_cash += exit_val
                        exit_price = close_val * (1.0 - self.slippage_penalty)
                    else:
                        exit_price_net = close_val * (1.0 + self.slippage_penalty)
                        pnl_pct = (pos['entry_price'] - exit_price_net) / pos['entry_price']
                        pnl_usd = pos['units'] * (pos['entry_price'] - exit_price_net)
                        shared_cash += (pos['units'] * pos['entry_price']) + pnl_usd
                        exit_price = exit_price_net
                    
                    self.trade_log.append({
                        'Ticker': ticker,
                        'EntryTime': pos['entry_time'],
                        'ExitTime': date,
                        'Direction': 'Long' if is_long else 'Short',
                        'EntryPrice': pos['entry_price'],
                        'ExitPrice': exit_price,
                        'PnL_Pct': pnl_pct,
                        'PnL_USD': pnl_usd,
                        'ExitReason': 'Emergency_Circuit_Breaker'
                    })
                    del positions[ticker]
                
                # Keep cash flat for remaining bars
                self.portfolio_equity[-1] = shared_cash
                continue

            # 4. EVALUATE NORMAL EXITS (SL / TP / SIGNAL EXITS)
            for ticker in list(positions.keys()):
                df = test_dfs[ticker]
                if date not in df.index:
                    continue
                
                pos = positions[ticker]
                high_val = df.loc[date, 'High']
                low_val = df.loc[date, 'Low']
                open_val = df.loc[date, 'Open']
                close_val = df.loc[date, 'Close']
                
                date_idx = df.index.get_loc(date)
                sig = test_signals_dict[ticker][date_idx]
                
                is_long = pos.get('direction', 'long') == 'long'
                
                if is_long:
                    stopped_out = low_val <= pos['sl']
                    target_hit = high_val >= pos['tp']
                else:
                    stopped_out = high_val >= pos['sl']
                    target_hit = low_val <= pos['tp']
                
                exit_triggered = False
                exit_price = 0.0
                reason = ""
                
                if stopped_out and target_hit:
                    stopped_out = True
                    target_hit = False
                
                if stopped_out:
                    exit_price = min(pos['sl'], open_val) if is_long else max(pos['sl'], open_val)
                    reason = "SL"
                    exit_triggered = True
                elif target_hit:
                    exit_price = max(pos['tp'], open_val) if is_long else min(pos['tp'], open_val)
                    reason = "TP"
                    exit_triggered = True
                elif (is_long and sig == -1) or (not is_long and sig == 1):
                    exit_price = close_val
                    reason = "Signal_Exit"
                    exit_triggered = True
                    
                if exit_triggered:
                    # Apply slippage penalty
                    if is_long:
                        slippage_fee = exit_price * self.slippage_penalty
                        final_exit_price = exit_price - slippage_fee
                        pnl_pct = (final_exit_price - pos['entry_price']) / pos['entry_price']
                        pnl_usd = pos['units'] * (final_exit_price - pos['entry_price'])
                        shared_cash += pos['units'] * final_exit_price
                    else:
                        slippage_fee = exit_price * self.slippage_penalty
                        final_exit_price = exit_price + slippage_fee
                        pnl_pct = (pos['entry_price'] - final_exit_price) / pos['entry_price']
                        pnl_usd = pos['units'] * (pos['entry_price'] - final_exit_price)
                        shared_cash += (pos['units'] * pos['entry_price']) + pnl_usd
                    
                    self.trade_log.append({
                        'Ticker': ticker,
                        'EntryTime': pos['entry_time'],
                        'ExitTime': date,
                        'Direction': 'Long' if is_long else 'Short',
                        'EntryPrice': pos['entry_price'],
                        'ExitPrice': final_exit_price,
                        'PnL_Pct': pnl_pct,
                        'PnL_USD': pnl_usd,
                        'ExitReason': f"{reason}_with_Slippage"
                    })
                    
                    # Only notify on the final day of the run (live signal today)
                    if date == all_dates[-1]:
                        send_push_notification(
                            f"🔴 **[EXIT]** Closed {'Long' if is_long else 'Short'} position on **{ticker}** at {final_exit_price:.2f} due to {reason}.\n"
                            f"PnL: {pnl_pct:.2%} (${pnl_usd:.2f})"
                        )
                    del positions[ticker]
                else:
                    # Trailing & Breakeven updates
                    if self.enable_breakeven and not pos['breakeven']:
                        if is_long:
                            if high_val >= (pos['entry_price'] + pos['entry_atr']):
                                positions[ticker]['sl'] = pos['entry_price']
                                positions[ticker]['breakeven'] = True
                        else:
                            if low_val <= (pos['entry_price'] - pos['entry_atr']):
                                positions[ticker]['sl'] = pos['entry_price']
                                positions[ticker]['breakeven'] = True
            
            # 5. EVALUATE NEW ENTRIES WITH CROSS-ASSET RELATIVE STRENGTH RANKING
            cash_at_start = current_equity  # Position size is based on total account equity
            
            candidates = []
            for ticker in test_dfs.keys():
                df = test_dfs[ticker]
                if date not in df.index or ticker in positions:
                    continue
                
                date_idx = df.index.get_loc(date)
                sig = test_signals_dict[ticker][date_idx]
                scale = test_allowance_dict[ticker][date_idx]
                
                if (sig == 1 or sig == -1) and scale > 0.0:
                    prob = 0.5
                    if test_probs_dict is not None and ticker in test_probs_dict:
                        prob = test_probs_dict[ticker][date_idx]
                    
                    # Compute entry strength (higher probability for long, lower for short)
                    strength = prob if sig == 1 else (1.0 - prob)
                    candidates.append({
                        'ticker': ticker,
                        'prob': prob,
                        'strength': strength,
                        'sig': sig,
                        'scale': scale,
                        'df': df,
                        'date_idx': date_idx
                    })
            
            # Sort candidates by strength in descending order (relative strength ranking)
            candidates = sorted(candidates, key=lambda x: x['strength'], reverse=True)
            
            # Execute only the single highest strength candidate to avoid capital dilution
            if candidates:
                best_cand = candidates[0]
                ticker = best_cand['ticker']
                df = best_cand['df']
                scale = best_cand['scale']
                sig = best_cand['sig']
                date_idx = best_cand['date_idx']
                
                close_val = df.loc[date, 'Close']
                atr_val = df.loc[date, 'ATR']
                
                # Position sizing
                allocation_usd = cash_at_start * self.max_alloc * scale
                
                if shared_cash >= allocation_usd and allocation_usd > 0:
                    tp_factor = 0.7 + 0.5 * scale  # maps scale=0 -> 0.7, scale=1 -> 1.2
                    sl_factor = 0.8 + 0.2 * scale  # maps scale=0 -> 0.8, scale=1 -> 1.0
                    
                    if sig == 1:
                        # Enter Long
                        entry_price = close_val * (1.0 + self.slippage_penalty)
                        units = allocation_usd / entry_price
                        shared_cash -= allocation_usd
                        
                        positions[ticker] = {
                            'units': units,
                            'entry_price': entry_price,
                            'entry_time': date,
                            'direction': 'long',
                            'sl': entry_price - (self.sl_mult_long * sl_factor * atr_val),
                            'tp': entry_price + (self.tp_mult * tp_factor * atr_val),
                            'entry_atr': atr_val,
                            'breakeven': False
                        }
                    else:
                        # Enter Short
                        entry_price = close_val * (1.0 - self.slippage_penalty)
                        units = allocation_usd / entry_price
                        shared_cash -= allocation_usd
                        
                        positions[ticker] = {
                            'units': units,
                            'entry_price': entry_price,
                            'entry_time': date,
                            'direction': 'short',
                            'sl': entry_price + (self.sl_mult_short * sl_factor * atr_val),
                            'tp': entry_price - (self.tp_mult * tp_factor * atr_val),
                            'entry_atr': atr_val,
                            'breakeven': False
                        }
                    
                    # Only notify on the final day of the run
                    if date == all_dates[-1]:
                        send_push_notification(
                            f"🟢 **[ENTRY]** Enter {'Long' if sig == 1 else 'Short'} position on **{ticker}** (AI Confidence: {best_cand['prob']:.1%}) at {entry_price:.2f}.\n"
                            f"SL: {positions[ticker]['sl']:.2f}, TP: {positions[ticker]['tp']:.2f} (Regime scale: {scale:.2f})"
                        )
                        
            # Update equity curve index point (re-calculate with actual entries/exits)
            current_equity = shared_cash
            for ticker in positions:
                df = test_dfs[ticker]
                pos = positions[ticker]
                is_long = pos.get('direction', 'long') == 'long'
                if date in df.index:
                    curr_price = df.loc[date, 'Close']
                else:
                    curr_price = pos['entry_price']
                
                if is_long:
                    current_equity += pos['units'] * curr_price
                else:
                    current_equity += (pos['units'] * pos['entry_price']) + (pos['units'] * (pos['entry_price'] - curr_price))
            self.portfolio_equity[-1] = current_equity
 
        # Force close any open positions at the end of the simulation
        final_date = all_dates[-1]
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            close_val = test_dfs[ticker].loc[final_date, 'Close']
            is_long = pos.get('direction', 'long') == 'long'
            
            if is_long:
                final_exit = close_val * (1.0 - self.slippage_penalty)
                pnl_pct = (final_exit - pos['entry_price']) / pos['entry_price']
                pnl_usd = pos['units'] * (final_exit - pos['entry_price'])
                shared_cash += pos['units'] * final_exit
            else:
                final_exit = close_val * (1.0 + self.slippage_penalty)
                pnl_pct = (pos['entry_price'] - final_exit) / pos['entry_price']
                pnl_usd = pos['units'] * (pos['entry_price'] - final_exit)
                shared_cash += (pos['units'] * pos['entry_price']) + pnl_usd
            
            self.trade_log.append({
                'Ticker': ticker,
                'EntryTime': pos['entry_time'],
                'ExitTime': final_date,
                'Direction': 'Long' if is_long else 'Short',
                'EntryPrice': pos['entry_price'],
                'ExitPrice': final_exit,
                'PnL_Pct': pnl_pct,
                'PnL_USD': pnl_usd,
                'ExitReason': 'End_Of_Backtest_with_Slippage'
            })
            del positions[ticker]
            
        self.portfolio_equity[-1] = shared_cash
 
        return pd.Series(self.portfolio_equity, index=all_dates), pd.DataFrame(self.trade_log)

    def analyze_performance(self, equity_series, trade_df, test_dfs):
        """Calculates standard portfolio metrics and outputs report."""
        final_equity = equity_series.iloc[-1]
        strategy_return = (final_equity - self.initial_capital) / self.initial_capital
        
        peaks = equity_series.cummax()
        drawdowns = (peaks - equity_series) / peaks
        max_dd = drawdowns.max()
        
        daily_returns = equity_series.pct_change().dropna()
        sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0.0
        
        bh_returns = []
        for ticker, df in test_dfs.items():
            bh_ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            bh_returns.append(bh_ret)
        avg_bh_return = np.mean(bh_returns) if bh_returns else 0.0
        
        num_trades = len(trade_df)
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        
        if num_trades > 0:
            winning = trade_df[trade_df['PnL_Pct'] > 0]
            losing = trade_df[trade_df['PnL_Pct'] <= 0]
            win_rate = len(winning) / num_trades
            avg_trade_pnl = trade_df['PnL_Pct'].mean()
            
            gross_prof = winning['PnL_USD'].sum()
            gross_loss = abs(losing['PnL_USD'].sum())
            profit_factor = gross_prof / gross_loss if gross_loss > 0 else float('inf')
            
        # Atomic Write
        if num_trades > 0:
            csv_content = trade_df.to_csv(index=False)
            safe_atomic_write("executed_trades.csv", csv_content)
            logger.info("Trade log written atomically to 'executed_trades.csv'.")

        logger.info("\n" + "="*50)
        logger.info("            PORTFOLIO PERFORMANCE RESULTS            ")
        logger.info("="*50)
        logger.info(f"Initial Capital:      ${self.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${final_equity:,.2f}")
        logger.info(f"Strategy Return:      {strategy_return:.2%}")
        logger.info(f"Average Market B&H:   {avg_bh_return:.2%}")
        logger.info(f"Outperformance:       {strategy_return - avg_bh_return:.2%}")
        logger.info(f"Max Portfolio DD:     {max_dd:.2%}")
        logger.info(f"Portfolio Sharpe:     {sharpe:.2f}")
        logger.info(f"Circuit Breaker Tripped: {self.circuit_breaker_tripped} {('on ' + str(self.circuit_breaker_date.date())) if self.circuit_breaker_tripped else ''}")
        logger.info("-"*50)
        logger.info(f"Total Trades Taken:   {num_trades}")
        logger.info(f"Win Rate:             {win_rate:.2%}")
        logger.info(f"Profit Factor:        {profit_factor:.2f}")
        logger.info(f"Average PnL / Trade:  {avg_trade_pnl:.2%}")
        logger.info("="*50)
        
        return {
            'strategy_return': strategy_return,
            'bh_return': avg_bh_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': num_trades,
            'win_rate': win_rate
        }
