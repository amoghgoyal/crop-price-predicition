import requests, time

# Test Open-Meteo
params = {'latitude':8.52,'longitude':76.94,'current':'temperature_2m,precipitation','timezone':'Asia/Kolkata','forecast_days':1}
try:
    s=time.time()
    r=requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=(10,30))
    print(f'Open-Meteo: {r.status_code} in {time.time()-s:.1f}s')
    if r.status_code==200:
        d=r.json()
        cc=d.get('current',{})
        print(f'  Temp: {cc.get("temperature_2m")}C, Precip: {cc.get("precipitation")}mm')
except Exception as e:
    print(f'Open-Meteo: {str(e)[:100]}')

# Test wttr.in (free, no key)
try:
    s=time.time()
    r=requests.get('https://wttr.in/Kerala?format=j1', timeout=(10,30))
    print(f'wttr.in: {r.status_code} in {time.time()-s:.1f}s')
    if r.status_code==200:
        d=r.json()
        cc=d.get('current_condition',[{}])[0]
        print(f'  Temp: {cc.get("temp_C")}C, Rain: {cc.get("precipMM")}mm, Humidity: {cc.get("humidity")}%')
        print(f'  Desc: {cc.get("weatherDesc",[{}])[0].get("value")}')
except Exception as e:
    print(f'wttr.in: {str(e)[:100]}')

# Test wttr.in with Delhi
try:
    s=time.time()
    r=requests.get('https://wttr.in/Delhi?format=j1', timeout=(10,30))
    print(f'wttr.in Delhi: {r.status_code} in {time.time()-s:.1f}s')
    if r.status_code==200:
        d=r.json()
        cc=d.get('current_condition',[{}])[0]
        print(f'  Temp: {cc.get("temp_C")}C, Rain: {cc.get("precipMM")}mm')
except Exception as e:
    print(f'wttr.in Delhi: {str(e)[:100]}')
