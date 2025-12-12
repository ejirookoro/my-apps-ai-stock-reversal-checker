import yfinance as yf

for sym in ['^IXIC','^DJI']:
    df = yf.download(sym, period='6mo', progress=False)
    print('\nSYM:', sym)
    if df is None or df.empty:
        print('None or empty')
        continue
    print('shape:', df.shape)
    print('head:\n', df.head().to_string())
    print('index dtype:', df.index.dtype)
    print('columns:', list(df.columns))
