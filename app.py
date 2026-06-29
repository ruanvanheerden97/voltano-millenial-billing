python3 -c "
import pandas as pd
xl = pd.ExcelFile('MIL_Battery_readings_EMS.xlsx')
d = pd.read_excel(xl, sheet_name=0)
d['Date'] = pd.to_datetime(d['Date'])
june25_29 = d[(d['Date'] >= '2026-06-25') & (d['Date'] <= '2026-06-29')]
print(june25_29[['Date','Solar Production Energy (kWh)','Battery Charge Energy (kWh)','Grid Imported Energy (kWh)']].to_string(index=False))
"