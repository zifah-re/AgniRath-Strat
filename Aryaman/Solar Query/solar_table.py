from datetime import datetime
from geopy.distance import geodesic
import numpy as np
import re

class SolarResultProxy:
    """A ultra-lightweight proxy to mimic the .data() access method without class overhead."""
    __slots__ = ('_data',)
    def __init__(self, data_dict):
        self._data = data_dict
    def data(self, key):
        if hasattr(key, '__iter__') and not isinstance(key, (str, bytes)):
            return [self._data[val] for val in key]
        return self._data[key]
    def __repr__(self):
        return str(self._data)


class SolarIrradiance:
    def __init__(self, data: dict | list, time_key="period_end", interval="PT5M"):
        if isinstance(data, list):
            tmp = {}
            for val in data:
                tmp[(val["lat"], val["lon"])] = val["data"]
            data = tmp
            
        # --- FIX: Handle interval if it is already an integer ---
        if isinstance(interval, (int, float)):
            self._interval = int(interval)
        else:
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', interval)
            if not match:
                raise ValueError(f"Invalid ISO 8601 duration string: {interval}")
            hours, minutes, seconds = match.groups()
            self._interval = ((int(hours) if hours else 0) * 3600 + 
                              (int(minutes) if minutes else 0) * 60 + 
                              (int(seconds) if seconds else 0))
        # --------------------------------------------------------
        
        self._data = data
        self._coords = list(data.keys())
        self._time_key = time_key
        raw_times = [i[self._time_key] for i in data[self._coords[0]]]
        if isinstance(raw_times[0], (int, float)):
            self._tz = datetime.now().astimezone().tzinfo
            self._time_list = np.array(raw_times, dtype=np.float64)
        elif isinstance(raw_times[0], str):
            self._tz = datetime.fromisoformat(raw_times[0]).tzinfo
            self._time_list = np.array([datetime.fromisoformat(i).timestamp() for i in raw_times], dtype=np.float64)
        else:
            raise TypeError(f"Invalid format for time: {type(raw_times[0]).__name__}")
            
        self._t0 = self._time_list[0]
        self._metrics = {}
        if self._coords:
            first_set = data[self._coords[0]]
            self._metric_keys = [k for k in first_set[0].keys() if k != time_key]
            
            for metric in self._metric_keys:
                matrix = []
                for coord in self._coords:
                    matrix.append([entry[metric] for entry in data[coord]])
                self._metrics[metric] = np.array(matrix, dtype=np.float64)

    def _get_closest_coord_idx(self, target_coord):
        if len(self._coords) <= 1:
            return 0
        dist = [geodesic(target_coord, c).kilometers for c in self._coords]
        return np.argmin(dist)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and not isinstance(key[0], (float, int)):
            x, y = key
        else:
            x, y = key, None

        if y is not None:
            if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                coord_val, time_val = x, y
            else:
                coord_val, time_val = y, x
            
            t_target = datetime.fromisoformat(time_val).timestamp() if isinstance(time_val, str) else float(time_val)
            coord_idx = self._get_closest_coord_idx(coord_val)
            
            n = int((t_target - self._t0) // self._interval)
            n = max(0, min(n, len(self._time_list) - 2)) # Safety bound check
            
            t1, t2 = self._time_list[n], self._time_list[n+1]
            weight = (t_target - t1) / (t2 - t1) if t2 != t1 else 0.0
            
            coord_results = {}
            for metric in self._metric_keys:
                y1 = self._metrics[metric][coord_idx, n]
                y2 = self._metrics[metric][coord_idx, n+1]
                coord_results[metric] = y1 + weight * (y2 - y1) # Fast linear interpolation
                
            return SolarResultProxy(coord_results)

        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            idx = self._get_closest_coord_idx(x)
            closest_coord = self._coords[idx]
            return SolarIrradiance({closest_coord: self._data[closest_coord]}, time_key=self._time_key,interval=self._interval)
            
        elif isinstance(x, (int, float, str)):
            t_target = datetime.fromisoformat(x).timestamp() if isinstance(x, str) else float(x)
            n = int((t_target - self._t0) // self._interval)
            n = max(0, min(n, len(self._time_list) - 2))
            
            t1, t2 = self._time_list[n], self._time_list[n+1]
            weight = (t_target - t1) / (t2 - t1) if t2 != t1 else 0.0
            
            output = {}
            for idx, coord in enumerate(self._coords):
                coord_results = {}
                for metric in self._metric_keys:
                    y1 = self._metrics[metric][idx, n]
                    y2 = self._metrics[metric][idx, n+1]
                    coord_results[metric] = float(y1 + weight * (y2 - y1))
                coord_results[self._time_key] = datetime.fromtimestamp(t_target, tz=self._tz).isoformat()
                output[coord] = [coord_results]
            return SolarIrradiance(output, time_key=self._time_key,interval=self._interval)

    def __repr__(self):
        return str(self._data)

    def data(self, key):
        if len(self._coords) == 1 and len(self._time_list) == 1:
            if hasattr(key, '__iter__') and not isinstance(key, (str, bytes)):
                return [self._data[self._coords[0]][0][val] for val in key]
            return self._data[self._coords[0]][0][key]