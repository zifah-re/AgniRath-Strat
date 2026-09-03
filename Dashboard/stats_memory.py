from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime
from typing import Any

BASE_DIR=Path(__file__).resolve().parent
LOGS_DIR=BASE_DIR/'Logs'
MAX_GAP_SECONDS=120.0


def f(v, default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except Exception:return default

def ts(packet):
    raw=packet.get('_rx_time') or packet.get('Timestamp')
    if isinstance(raw,(int,float)):
        try:return datetime.fromtimestamp(float(raw))
        except Exception:return None
    if not raw:return None
    try:return datetime.fromisoformat(str(raw).replace('Z','+00:00'))
    except Exception:return None

def solar_power(p):
    # Same definition used by main.py: sum MPPT output power.
    vals=[]
    for c in 'ABCD':
        vk=f'Output_Voltage_{c}'; ik=f'Output_Current_{c}'
        if vk in p and ik in p: vals.append(f(p[vk])*f(p[ik]))
    return sum(vals) if vals else None

def bus_power(p):
    if 'Bus_Voltage' not in p or 'Bus_Current' not in p:return None
    return abs(f(p['Bus_Voltage'])*f(p['Bus_Current']))

def velocity(p):return max(0.0,f(p.get('Vehicle_Velocity'))) # m/s

class StatsMemory:
    def __init__(self):
        self.runs=[]
        self.loaded_files=[]
        self.live_packets=[]
        self.live_capture=False

    def _read(self,path):
        out=[]
        try:
            for line in path.open('r',encoding='utf-8'):
                try:
                    p=json.loads(line)
                    if isinstance(p,dict) and ts(p):out.append(p)
                except Exception:pass
        except Exception:pass
        return sorted(out,key=lambda p:ts(p))

    def available_logs(self):
        ans=[]
        for p in LOGS_DIR.glob('*.jsonl'):
            packets=self._read(p)
            if not packets:continue
            a,b=ts(packets[0]),ts(packets[-1])
            ans.append({'filename':p.name,'packet_count':len(packets),'start_time':a.isoformat(),'end_time':b.isoformat(),'duration_seconds':max(0,(b-a).total_seconds())})
        return sorted(ans,key=lambda x:x['start_time'])

    def load_runs(self,names):
        allowed={Path(str(x)).name for x in names}
        runs=[]
        for p in LOGS_DIR.glob('*.jsonl'):
            if p.name not in allowed:continue
            packets=self._read(p)
            if packets:runs.append({'filename':p.name,'packets':packets,'start':ts(packets[0])})
        self.runs=sorted(runs,key=lambda r:r['start'])
        self.loaded_files=[r['filename'] for r in self.runs]
        self.live_packets=[]
        return {'success':True,'loaded_files':self.loaded_files,'run_count':len(self.runs)}

    def start_live(self):
        self.live_packets=[]; self.live_capture=True
    def stop_live(self):self.live_capture=False
    def clear_live(self):self.live_packets=[]
    def add_live_packet(self,p):
        if self.live_capture and isinstance(p,dict) and ts(p):self.live_packets.append(dict(p))

    def process(self,name,packets):
        if not packets:return {'filename':name,'timeline':[],'stats':{}}
        d=e_in=e_out=0.0; active=0.0
        maxv=maxrpm=0.0; speeds=[]; temps={}; soc=[]
        timeline=[]; prev=None; offset=0.0
        for p in packets:
            t=ts(p)
            if not t:continue
            v=velocity(p); maxv=max(maxv,v); speeds.append(v)
            maxrpm=max(maxrpm,abs(f(p.get('Motor_Velocity')))*3.6)
            if 'SOC_Ah' in p:soc.append(f(p['SOC_Ah']))
            for k,vv in p.items():
                if 'Temp' in k or 'Temperature' in k:
                    x=f(vv,float('nan'))
                    if math.isfinite(x) and -50<=x<=200:temps[k]=max(temps.get(k,-1e9),x)
            if prev:
                dt=(t-ts(prev)).total_seconds()
                if 0<dt<=MAX_GAP_SECONDS:
                    active+=dt
                    d+=((velocity(prev)+v)/2)*dt
                    sp1,sp2=solar_power(prev),solar_power(p)
                    if sp1 is not None and sp2 is not None:e_in+=((sp1+sp2)/2)*dt/3600
                    bp1,bp2=bus_power(prev),bus_power(p)
                    if bp1 is not None and bp2 is not None:e_out+=((bp1+bp2)/2)*dt/3600
            timeline.append({'timestamp':t.isoformat(),'run_time_seconds':offset,'velocity_kmh':v*3.6,'solar_power_w':solar_power(p),'bus_power_w':bus_power(p),'cumulative_distance_km':d/1000,'cumulative_energy_received_wh':e_in,'cumulative_energy_lost_wh':e_out})
            if prev:offset+=(max(0,(t-ts(prev)).total_seconds()))
            prev=p
        st={'distance_km':d/1000,'energy_received_wh':e_in,'energy_lost_wh':e_out,'net_energy_wh':e_in-e_out,'active_time_seconds':active,'max_velocity_kmh':maxv*3.6,'average_velocity_kmh':(sum(speeds)/len(speeds)*3.6 if speeds else 0),'max_motor_velocity_kmh':maxrpm,'initial_soc_ah':soc[0] if soc else None,'final_soc_ah':soc[-1] if soc else None,'min_soc_ah':min(soc) if soc else None,'max_soc_ah':max(soc) if soc else None,'max_temperatures':temps}
        return {'filename':name,'start_time':ts(packets[0]).isoformat(),'end_time':ts(packets[-1]).isoformat(),'stats':st,'timeline':timeline}

    def build(self):
        processed=[self.process(r['filename'],r['packets']) for r in self.runs]
        live=self.process('LIVE_SESSION',self.live_packets) if self.live_packets else None
        if live:processed.append(live)
        totals={'cumulative_distance_km':0.0,'cumulative_energy_received_wh':0.0,'cumulative_energy_lost_wh':0.0,'total_active_time_seconds':0.0,'maximum_velocity_kmh':0.0}
        combined=[]
        for run in processed:
            s=run['stats']; base_d=totals['cumulative_distance_km']; base_in=totals['cumulative_energy_received_wh']; base_out=totals['cumulative_energy_lost_wh']
            for p in run['timeline']:
                q=dict(p);q['run']=run['filename'];q['cumulative_distance_km']=base_d+p['cumulative_distance_km'];q['cumulative_energy_received_wh']=base_in+p['cumulative_energy_received_wh'];q['cumulative_energy_lost_wh']=base_out+p['cumulative_energy_lost_wh'];combined.append(q)
            totals['cumulative_distance_km']+=s.get('distance_km',0);totals['cumulative_energy_received_wh']+=s.get('energy_received_wh',0);totals['cumulative_energy_lost_wh']+=s.get('energy_lost_wh',0);totals['total_active_time_seconds']+=s.get('active_time_seconds',0);totals['maximum_velocity_kmh']=max(totals['maximum_velocity_kmh'],s.get('max_velocity_kmh',0))
        totals['net_energy_wh']=totals['cumulative_energy_received_wh']-totals['cumulative_energy_lost_wh']
        return {'loaded_files':self.loaded_files,'historical_run_count':len(self.runs),'live_capture':self.live_capture,'live_packet_count':len(self.live_packets),'overall':totals,'runs':processed,'live_run':live,'timeline':combined}

stats_memory=StatsMemory()
