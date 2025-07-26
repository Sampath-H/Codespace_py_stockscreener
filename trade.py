
def get_weekdays_since_friday(friday_date):
    """Return a list of weekdays (dates) since the given Friday up to today (excluding weekends)."""
    today = datetime.now().date()
    days = []
    current = friday_date + timedelta(days=1)
    while current <= today:
        if current.weekday() < 5:  # Monday=0, ..., Friday=4
            days.append(current)
        current += timedelta(days=1)
    return days
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io
import base64
import os
from email_alert import send_email_alert


# Set page configuration
st.set_page_config(
    page_title="Friday Breakout Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data
def get_last_friday():
    """Calculate the date of the last Friday"""

    today = datetime.now()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset + 7 if offset == 0 and today.hour < 18 else offset)
    return last_friday.date()
        


def get_friday_first_hour_cluster(symbol, friday_date):
    """Get Friday's first hour high and low (approximated using day's open and high/low)"""
    try:
        # For NSE, we'll approximate first hour cluster as the range between open and 
        # the highest/lowest point reached in early trading
        # Since we don't have intraday data, we'll use a heuristic:
        # First hour cluster = Open ± 1% or the day's range if smaller
        
        data = yf.download(symbol, start=friday_date, end=friday_date + timedelta(days=1), progress=False, auto_adjust=True)
        if data is None or len(data) == 0:
            return None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.reset_index()
        data['Date'] = data['Date'].dt.date
        friday_row = data[data['Date'] == friday_date]
        if len(friday_row) == 0:
            return None, None
        friday_row = friday_row.iloc[0]
        open_price = float(friday_row['Open'])
        day_high = float(friday_row['High'])
        day_low = float(friday_row['Low'])
        # Heuristic for first hour cluster: 
        # Use smaller range between 1% of open price or 50% of day's range
        one_percent_range = open_price * 0.01
        half_day_range = (day_high - day_low) * 0.5
        cluster_range = min(one_percent_range, half_day_range)
        # First hour cluster bounds
        cluster_high = open_price + cluster_range
        cluster_low = open_price - cluster_range
        # Ensure cluster is within day's actual range
        cluster_high = min(cluster_high, day_high)
        cluster_low = max(cluster_low, day_low)
        return cluster_high, cluster_low
        
    except Exception as e:
        st.warning(f"Error getting Friday cluster for {symbol}: {e}")
        return None, None



def fetch_data(symbols, progress_bar=None, analysis_type="basic"):
    """Fetch stock data and perform Friday breakout analysis"""
    results = []
    last_friday = get_last_friday()
    start_date = last_friday - timedelta(days=7)
    end_date = datetime.now().date()
    
    total_symbols = len(symbols)
    
    for i, symbol in enumerate(symbols):
        try:
            # Update progress bar
            if progress_bar:
                progress_bar.progress((i + 1) / total_symbols, 
                                    text=f"Processing {symbol} ({i + 1}/{total_symbols})")
            
            # Download stock data with proper error handling
            data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)
            # Check if data exists
            if data is None or len(data) == 0:
                continue
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten the multi-level columns
                data.columns = [col[0] for col in data.columns]
            # Reset index to work with datetime index properly
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date
            
            # Get Friday's data using boolean indexing properly
            friday_mask = (data['Date'] == last_friday)
            friday_data = data[friday_mask]
            
            if len(friday_data) == 0:
                continue

            # Get latest trading data (most recent row)
            latest_row = data.iloc[-1]
            latest_date = latest_row['Date']
            
            # Calculate previous close (for change calculation)
            if len(data) >= 2:
                prev_close = data.iloc[-2]['Close']
            else:
                prev_close = latest_row['Open']

            # Current price and changes
            latest_close = latest_row['Close']
            chng = latest_close - prev_close
            pct_chng = (chng / prev_close) * 100 if prev_close != 0 else 0

            # Friday high and low
            friday_low = friday_data['Low'].iloc[0]
            friday_high = friday_data['High'].iloc[0]

            # Initialize result dictionary
            result = {
                'Stock': symbol.replace('.NS', ''),
                'Latest Date': latest_date,
                'Open': format_price(latest_row['Open']),
                'High': format_price(latest_row['High']),
                'Low': format_price(latest_row['Low']),
                'Prev. Close': format_price(prev_close),
                'LTP': format_price(latest_close),
                'CHNG': format_price(chng),
                '%CHNG': format_price(pct_chng),
                'Friday High': format_price(friday_high),
                'Friday Low': format_price(friday_low)
            }

            # Determine signal based on analysis type
            if analysis_type == "cluster":
                # Enhanced analysis with Friday cluster logic
                signal, cluster_high, cluster_low = analyze_with_cluster_logic(
                    symbol, data, last_friday, friday_high, friday_low, latest_close
                )
                result['Signal'] = signal
                result['Friday Cluster High'] = format_price(cluster_high) if cluster_high else 'N/A'
                result['Friday Cluster Low'] = format_price(cluster_low) if cluster_low else 'N/A'
            else:
                # Basic analysis
                if latest_close > friday_high:
                    signal = 'Bullish Confirmed'
                elif latest_close < friday_low:
                    signal = 'Bearish Confirmed'
                else:
                    signal = 'Neutral'
                result['Signal'] = signal

            # ...database save removed...
            
            results.append(result)
            
        except Exception as e:
            st.warning(f"Error fetching {symbol}: {e}")
            continue
    
    return results

def analyze_with_cluster_logic(symbol, data, friday_date, friday_high, friday_low, current_price):
    """Analyze stock with Friday cluster return logic"""
    try:
        # Get Friday's first hour cluster
        cluster_high, cluster_low = get_friday_first_hour_cluster(symbol, friday_date)
        
        if cluster_high is None or cluster_low is None:
            # Fallback to basic analysis if cluster data unavailable
            if current_price > friday_high:
                return 'Bullish Confirmed', None, None
            elif current_price < friday_low:
                return 'Bearish Confirmed', None, None
            else:
                return 'Neutral', None, None
        
        # Get weekdays since Friday to track breakout/breakdown patterns
        weekdays = get_weekdays_since_friday(friday_date)
        
        if not weekdays:
            return 'Neutral', cluster_high, cluster_low
        
        # Track if there was a breakout/breakdown during the week
        had_breakout = False
        had_breakdown = False
        
        for day in weekdays:
            day_data = data[data['Date'] == day]
            if len(day_data) == 0:
                continue
                
            day_row = day_data.iloc[0]
            day_close = float(day_row['Close'])
            
            # Check if breakout occurred (CLOSE above Friday high or below Friday low)
            if day_close > friday_high:
                had_breakout = True
            if day_close < friday_low:
                had_breakdown = True
        
        # Current price analysis with cluster logic
        current_in_cluster = cluster_low <= current_price <= cluster_high
        
        # Debug logging for development
        if symbol.replace('.NS', '') in ['SUNPHARMA', 'SJVN']:  # Only for specific stocks mentioned
            st.write(f"Debug {symbol.replace('.NS', '')}: Current={current_price}, Friday High={friday_high}, Friday Low={friday_low}, Cluster={cluster_low}-{cluster_high}, InCluster={current_in_cluster}, Breakout={had_breakout}, Breakdown={had_breakdown}")
        
        # Priority order: Check cluster returns first, then current position
        if had_breakdown and current_in_cluster:
            return 'Breakdown Done but Price Returns Friday\'s Cluster', cluster_high, cluster_low
        elif had_breakout and current_in_cluster:
            return 'Breakout Done but Price Returns Friday\'s Cluster', cluster_high, cluster_low
        elif current_price > friday_high:
            return 'Bullish Confirmed', cluster_high, cluster_low
        elif current_price < friday_low:
            return 'Bearish Confirmed', cluster_high, cluster_low
        elif had_breakout or had_breakdown:
            # Had movement but now in different zone (not cluster, not breakout zone)
            return 'Post-Movement Consolidation', cluster_high, cluster_low
        else:
            return 'Neutral', cluster_high, cluster_low
            
    except Exception as e:
        st.warning(f"Cluster analysis error for {symbol}: {e}")
        # Fallback to basic analysis
        if current_price > friday_high:
            return 'Bullish Confirmed', None, None
        elif current_price < friday_low:
            return 'Bearish Confirmed', None, None
        else:
            return 'Neutral', None, None



def fetch_daily_breakout_data(symbols, progress_bar=None):
    """Fetch daily breakout data showing which stocks broke out on each day"""
    last_friday = get_last_friday()
    weekdays = get_weekdays_since_friday(last_friday)
    
    if not weekdays:
        return []
    
    start_date = last_friday - timedelta(days=7)
    end_date = datetime.now().date()
    
    daily_results = []
    total_symbols = len(symbols)
    
    for i, symbol in enumerate(symbols):
        try:
            # Update progress bar
            if progress_bar:
                progress_bar.progress((i + 1) / total_symbols, 
                                    text=f"Processing daily data for {symbol} ({i + 1}/{total_symbols})")
            
            # Download stock data directly (no database)
            data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)
            if data is None or len(data) == 0:
                continue
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            # Reset index and prepare data
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date
            
            # Get Friday's data
            friday_mask = (data['Date'] == last_friday)
            friday_data = data[friday_mask]
            
            if len(friday_data) == 0:
                continue

            friday_low = friday_data['Low'].iloc[0]
            friday_high = friday_data['High'].iloc[0]
            
            # Check breakout for each weekday
            breakout_day = None
            breakout_type = None
            
            for day in weekdays:
                day_mask = (data['Date'] == day)
                day_data = data[day_mask]
                
                if len(day_data) == 0:
                    continue
                
                day_high = day_data['High'].iloc[0]
                day_low = day_data['Low'].iloc[0]
                
                # Check if breakout occurred on this day
                if day_high > friday_high and breakout_day is None:
                    breakout_day = day
                    breakout_type = 'Bullish'
                    break
                elif day_low < friday_low and breakout_day is None:
                    breakout_day = day
                    breakout_type = 'Bearish'
                    break
            
            # Get current status
            latest_row = data.iloc[-1]
            latest_close = latest_row['Close']
            
            if latest_close > friday_high:
                current_signal = 'Bullish Confirmed'
            elif latest_close < friday_low:
                current_signal = 'Bearish Confirmed'
            else:
                current_signal = 'Neutral'
            
            daily_results.append({
                'Stock': symbol.replace('.NS', ''),
                'Friday High': format_price(friday_high),
                'Friday Low': format_price(friday_low),
                'Breakout Day': breakout_day.strftime('%A, %b %d') if breakout_day else 'No Breakout',
                'Breakout Type': breakout_type if breakout_type else 'None',
                'Current Price': format_price(latest_close),
                'Current Signal': current_signal,
                'Days Since Friday': len(weekdays) if weekdays else 0
            })
            
        except Exception as e:
            st.warning(f"Error fetching daily data for {symbol}: {e}")
            continue
    
    return daily_results

def color_signal(val):
    """Color code the signals"""
    if val == 'Bullish Confirmed':
        return 'background-color: #d4edda; color: #155724'
    elif val == 'Bearish Confirmed':
        return 'background-color: #f8d7da; color: #721c24'
    elif 'Breakout Done but Price Returns' in val:
        return 'background-color: #fff3cd; color: #856404'
    elif 'Breakdown Done but Price Returns' in val:
        return 'background-color: #f8d7da; color: #721c24; font-style: italic'
    elif val == 'Post-Movement Consolidation':
        return 'background-color: #cce5ff; color: #004085'
    else:
        return 'background-color: #e2e3e5; color: #383d41'

def format_price(price):
    """Format price to remove unnecessary decimal places"""
    try:
        price_float = float(price)
        # If the price is a whole number, show no decimals
        if price_float == int(price_float):
            return str(int(price_float))
        # Otherwise, show up to 2 decimal places, removing trailing zeros
        else:
            return f"{price_float:.2f}".rstrip('0').rstrip('.')
    except:
        return str(price)

def color_change(val):
    """Color code the price changes"""
    try:
        val_float = float(val)
        if val_float > 0:
            return 'color: #28a745'
        elif val_float < 0:
            return 'color: #dc3545'
        else:
            return 'color: #6c757d'
    except:
        return ''

def create_download_link(df, filename):
    """Create a download link for Excel file"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Screener Results')
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    return href

def main():
    st.title("📈 Friday Breakout Stock Screener")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # ...database status UI removed...
    st.sidebar.markdown("---")
    
    # File upload or default symbols, with F&O/Nifty 500 selector
    st.sidebar.markdown("### 📂 Stock Universe")
    stock_universe = st.sidebar.radio(
        "Select Stock Universe",
        ["Nifty 500", "F&O Stocks"],
        index=0
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with stock symbols",
        type=['csv'],
        help="Upload a CSV file with a 'Symbol' column containing stock symbols"
    )

    # Load symbols
    if uploaded_file is not None:
        try:
            symbols_df = pd.read_csv(uploaded_file)
            # Check for Symbol column (case insensitive)
            symbol_col = None
            for col in symbols_df.columns:
                if col.lower() in ['symbol', 'symbols']:
                    symbol_col = col
                    break
            if symbol_col is None:
                st.sidebar.error("CSV file must contain a 'Symbol' or 'SYMBOL' column")
                return
            symbols = symbols_df[symbol_col].tolist()
            st.sidebar.success(f"Loaded {len(symbols)} symbols from uploaded file")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
            return
    else:
        # Use default based on radio selection
        try:
            if stock_universe == "Nifty 500":
                symbols_df = pd.read_csv('stocks.csv')
                symbols = symbols_df['Symbol'].tolist()
                st.sidebar.info(f"Using default Nifty 500 symbols ({len(symbols)} stocks)")
            else:
                symbols_df = pd.read_csv('NSE_FO_Stocks_NS.csv')
                # Accept both 'SYMBOL' and 'Symbol' as column name
                symbol_col = None
                for col in symbols_df.columns:
                    if col.lower() in ['symbol', 'symbols']:
                        symbol_col = col
                        break
                if symbol_col is None:
                    st.sidebar.error("F&O CSV file must contain a 'Symbol' or 'SYMBOL' column")
                    return
                symbols = symbols_df[symbol_col].tolist()
                st.sidebar.info(f"Using default F&O symbols ({len(symbols)} stocks)")
        except FileNotFoundError:
            st.sidebar.error("Default stocks file not found. Please upload a CSV file.")
            return
        except Exception as e:
            st.sidebar.error(f"Error reading default stocks file: {e}")
            return
    

    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Current Signals", "Current Signals with Cluster Analysis", "Daily Breakout Tracking", "Both"]
    )

    # Email alert option
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Email Alert Settings")
    enable_email_alert = st.sidebar.checkbox("Enable Email Alert when price returns to Friday's cluster")
    user_email = None
    if enable_email_alert:
        user_email = st.sidebar.text_input("Enter your Gmail address for alerts", value="sampathskh@gmail.com")
        st.sidebar.info("You will receive an email alert if any stock returns to Friday's cluster.")

    # Analysis method for current signals
    analysis_method = "basic"
    if analysis_type in ["Current Signals with Cluster Analysis", "Both"]:
        analysis_method = "cluster"

    # Signal filter - dynamic based on analysis type
    if analysis_method == "cluster":
        signal_options = [
            "Bullish Confirmed", 
            "Bearish Confirmed", 
            "Breakout Done but Price Returns Friday's Cluster",
            "Breakdown Done but Price Returns Friday's Cluster",
            "Post-Movement Consolidation",
            "Neutral"
        ]
        default_signals = [
            "Bullish Confirmed", 
            "Bearish Confirmed",
            "Breakout Done but Price Returns Friday's Cluster",
            "Breakdown Done but Price Returns Friday's Cluster"
        ]
    else:
        signal_options = ["Bullish Confirmed", "Bearish Confirmed", "Neutral"]
        default_signals = ["Bullish Confirmed", "Bearish Confirmed", "Neutral"]

    signal_filter = st.sidebar.multiselect(
        "Filter by Signal",
        signal_options,
        default=default_signals
    )

    # Display last Friday info
    last_friday = get_last_friday()
    weekdays = get_weekdays_since_friday(last_friday)

    st.info(f"📅 Reference Friday: {last_friday.strftime('%A, %B %d, %Y')} | Trading days since: {len(weekdays)}")
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not symbols:
            st.error("No symbols to analyze")
            return

        # Current signals analysis
        if analysis_type in ["Current Signals", "Current Signals with Cluster Analysis", "Both"]:
            if analysis_method == "cluster":
                st.subheader("📊 Current Trading Signals with Friday Cluster Analysis")
                st.info("🔍 This analysis detects when stocks return to Friday's first-hour trading cluster after initial breakouts/breakdowns.")

                # Add legend for cluster analysis
                with st.expander("📋 Signal Meanings"):
                    st.markdown("""
                    **Signal Types:**
                    - **Bullish Confirmed**: Price above Friday's high and staying strong
                    - **Bearish Confirmed**: Price below Friday's low and staying weak  
                    - **Breakout Done but Returns Friday's Cluster**: Stock broke above Friday's high during the week but current price has returned to Friday's first-hour trading range
                    - **Breakdown Done but Returns Friday's Cluster**: Stock broke below Friday's low during the week but current price has returned to Friday's first-hour trading range
                    - **Post-Movement Consolidation**: Had significant movement during week but now consolidating in middle zones
                    - **Neutral**: No significant breakout or breakdown occurred during the week

                    **Friday's Cluster**: Approximated as the first-hour trading range around Friday's opening price, representing the initial price discovery zone.
                    """)
            else:
                st.subheader("📊 Current Trading Signals")

            progress_bar = st.progress(0, text="Initializing...")
            results = fetch_data(symbols, progress_bar, analysis_method)
            progress_bar.empty()

            # Email alert logic
            alert_sent = False
            if enable_email_alert and user_email:
                # Check for any cluster return signals
                for res in results:
                    if res.get('Signal') in [
                        "Breakout Done but Price Returns Friday's Cluster",
                        "Breakdown Done but Price Returns Friday's Cluster"
                    ]:
                        send_email_alert(user_email, res['Stock'], res['Signal'], res.get('LTP'), res.get('Friday Cluster High'), res.get('Friday Cluster Low'))
                        alert_sent = True
                if alert_sent:
                    st.success(f"Email alert sent to {user_email} for stocks returning to Friday's cluster.")
                else:
                    st.info("No stocks triggered the alert condition.")

            if results:
                df = pd.DataFrame(results)

                # Apply signal filter
                if signal_filter:
                    df = df[df['Signal'].isin(signal_filter)]

                # Search functionality
                search_term = st.text_input("🔍 Search stocks", placeholder="Enter stock symbol or name...")
                if search_term:
                    df = df[df['Stock'].str.contains(search_term.upper(), na=False)]

                if not df.empty:
                    if analysis_method == "cluster":
                        # Enhanced metrics for cluster analysis
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Stocks", len(df))
                        with col2:
                            cluster_return_count = len(df[df['Signal'].str.contains('Returns Friday\'s Cluster', na=False)])
                            st.metric("Cluster Returns", cluster_return_count)
                        with col3:
                            confirmed_count = len(df[df['Signal'].isin(['Bullish Confirmed', 'Bearish Confirmed'])])
                            st.metric("Strong Moves", confirmed_count)

                        # Additional row for detailed breakdown
                        col4, col5, col6, col7 = st.columns(4)
                        with col4:
                            bullish_count = len(df[df['Signal'] == 'Bullish Confirmed'])
                            st.metric("Bullish", bullish_count)
                        with col5:
                            bearish_count = len(df[df['Signal'] == 'Bearish Confirmed'])
                            st.metric("Bearish", bearish_count)
                        with col6:
                            breakout_return_count = len(df[df['Signal'] == 'Breakout Done but Price Returns Friday\'s Cluster'])
                            st.metric("Breakout Returns", breakout_return_count)
                        with col7:
                            breakdown_return_count = len(df[df['Signal'] == 'Breakdown Done but Price Returns Friday\'s Cluster'])
                            st.metric("Breakdown Returns", breakdown_return_count)
                    else:
                        # Basic metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Stocks", len(df))
                        with col2:
                            bullish_count = len(df[df['Signal'] == 'Bullish Confirmed'])
                            st.metric("Bullish Signals", bullish_count)
                        with col3:
                            bearish_count = len(df[df['Signal'] == 'Bearish Confirmed'])
                            st.metric("Bearish Signals", bearish_count)
                        with col4:
                            neutral_count = len(df[df['Signal'] == 'Neutral'])
                            st.metric("Neutral", neutral_count)

                    # Style the dataframe
                    styled_df = df.style.map(color_signal, subset=['Signal']) \
                                      .map(color_change, subset=['CHNG', '%CHNG'])

                    st.dataframe(styled_df, use_container_width=True)

                    # Download button
                    filename = f"friday_breakout_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.markdown(create_download_link(df, filename), unsafe_allow_html=True)

                else:
                    st.warning("No stocks match the current filters")
            else:
                st.error("No data could be fetched. Please check your internet connection and try again.")
        
        # Daily breakout tracking
        if analysis_type in ["Daily Breakout Tracking", "Both"]:
            st.subheader("📈 Daily Breakout Tracking")
            
            progress_bar = st.progress(0, text="Fetching daily breakout data...")
            daily_results = fetch_daily_breakout_data(symbols, progress_bar)
            progress_bar.empty()
            
            if daily_results:
                daily_df = pd.DataFrame(daily_results)
                
                # Apply signal filter for current signal
                if signal_filter:
                    daily_df = daily_df[daily_df['Current Signal'].isin(signal_filter)]
                
                # Search functionality
                search_term_daily = st.text_input("🔍 Search stocks (Daily)", 
                                                placeholder="Enter stock symbol or name...", 
                                                key="daily_search")
                if search_term_daily:
                    daily_df = daily_df[daily_df['Stock'].str.contains(search_term_daily.upper(), na=False)]
                
                if not daily_df.empty:
                    # Display breakout metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        bullish_breakouts = len(daily_df[daily_df['Breakout Type'] == 'Bullish'])
                        st.metric("Bullish Breakouts", bullish_breakouts)
                    with col2:
                        bearish_breakouts = len(daily_df[daily_df['Breakout Type'] == 'Bearish'])
                        st.metric("Bearish Breakouts", bearish_breakouts)
                    with col3:
                        no_breakouts = len(daily_df[daily_df['Breakout Type'] == 'None'])
                        st.metric("No Breakouts", no_breakouts)
                    
                    # Style the daily dataframe
                    styled_daily_df = daily_df.style.map(color_signal, subset=['Current Signal'])
                    
                    st.dataframe(styled_daily_df, use_container_width=True)
                    
                    # Download button for daily data
                    daily_filename = f"daily_breakout_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.markdown(create_download_link(daily_df, daily_filename), unsafe_allow_html=True)
                    
                else:
                    st.warning("No stocks match the current filters for daily tracking")
            else:
                st.error("No daily breakout data could be fetched")
    
    # Information section
    with st.expander("ℹ️ How Friday Breakout Analysis Works"):
        st.markdown("""
        **Friday Breakout Strategy:**
        
        1. **Reference Point**: Every Friday's high and low prices serve as key levels
        2. **Bullish Signal**: When stock price breaks above Friday's high
        3. **Bearish Signal**: When stock price breaks below Friday's low
        4. **Neutral**: Price remains between Friday's high and low
        
        **Signal Types:**
        - 🟢 **Bullish Confirmed**: Current price > Friday High
        - 🔴 **Bearish Confirmed**: Current price < Friday Low  
        - ⚪ **Neutral**: Friday Low ≤ Current price ≤ Friday High
        
        **Daily Tracking**: Shows the exact day when breakout occurred since last Friday
        """)

if __name__ == "__main__":
    main()
