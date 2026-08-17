from datetime import datetime
from geopy.distance import geodesic
import numpy as np
import re

class SolarResultProxy:
    """Lightweight proxy exposing .data() over an array-valued metric dict."""
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
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', interval)
        if not match:
            raise ValueError("Invalid ISO 8601 duration string")
        hours, minutes, seconds = match.groups()
        self._interval = ((int(hours) if hours else 0) * 3600 + 
                          (int(minutes) if minutes else 0) * 60 + 
                          (int(seconds) if seconds else 0))
        
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

    def _to_epoch_array(self, t):
        """Always returns a 1D float64 epoch-seconds array, whatever t is."""
        arr = np.atleast_1d(t)
        if arr.dtype.kind in ('U', 'S', 'O'):
            return np.array(
                [datetime.fromisoformat(v).timestamp() if isinstance(v, str) else float(v) for v in arr],
                dtype=np.float64
            )
        return arr.astype(np.float64)

    def _interp_indices_and_weights(self, times):
        """Bucket index + interpolation weight for every entry of times at once."""
        n = np.floor((times - self._t0) / self._interval).astype(np.int64)
        n = np.clip(n, 0, len(self._time_list) - 2)
        t1 = self._time_list[n]
        t2 = self._time_list[n + 1]
        span = t2 - t1
        weight = np.divide(times - t1, span, out=np.zeros_like(times), where=span != 0)
        return n, weight

    def _lookup_vectorized(self, t):
        """Interpolate every metric, for every coord, across the full time array at once."""
        times = self._to_epoch_array(t)
        n, weight = self._interp_indices_and_weights(times)

        results = {}
        for idx, coord in enumerate(self._coords):
            coord_results = {}
            for metric in self._metric_keys:
                arr = self._metrics[metric][idx]
                coord_results[metric] = arr[n] + weight * (arr[n + 1] - arr[n])
            results[coord] = SolarResultProxy(coord_results)

        return results if len(results) > 1 else next(iter(results.values()))

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and not isinstance(key[0], (float, int)):
            x, y = key
        else:
            x, y = key, None

        if y is not None:
            # tuple => coordinate; everything else => time (vector or scalar)
            coord_val, time_val = (x, y) if isinstance(x, tuple) else (y, x)
            coord_idx = self._get_closest_coord_idx(coord_val)
            times = self._to_epoch_array(time_val)
            n, weight = self._interp_indices_and_weights(times)

            coord_results = {}
            for metric in self._metric_keys:
                arr = self._metrics[metric][coord_idx]
                coord_results[metric] = arr[n] + weight * (arr[n + 1] - arr[n])
            return SolarResultProxy(coord_results)

        # coordinate narrowing to the nearest station
        if isinstance(x, tuple):
            idx = self._get_closest_coord_idx(x)
            closest_coord = self._coords[idx]
            return SolarIrradiance({closest_coord: self._data[closest_coord]}, time_key=self._time_key, interval=self._interval)

        # everything else is a time lookup, always vectorized
        return self._lookup_vectorized(x)

    def __repr__(self):
        return str(self._data)