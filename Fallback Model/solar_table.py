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
    def __init__(self, data: dict | list, time_key="period_end", interval="PT5M", radius=None):
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
        self._coords = np.array(list(data.keys()))
        self._time_key = time_key
        self._radius = radius
        raw_times = [i[self._time_key] for i in data[tuple(self._coords[0])]]
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
        if len(self._coords) > 0:
            first_set = data[tuple(self._coords[0])]
            self._metric_keys = [k for k in first_set[0].keys() if k != time_key]
            
            for metric in self._metric_keys:
                matrix = []
                for coord in self._coords:
                    matrix.append([entry[metric] for entry in data[tuple(coord)]])
                self._metrics[metric] = np.array(matrix, dtype=np.float64)

    def _calculate_min_distance(self, coord):
        if len(self._coords) <= 1:
            return 0
        dist = []
        for i, c in enumerate(self._coords):
            x = geodesic(coord, c).kilometers
            if self._radius is not None and x < self._radius:
                return i
            dist.append(x)
        return int(np.argmin(dist))

    def _get_closest_coord_idx(self, target_coord):
        arr = np.asarray(target_coord)
        if arr.ndim == 1 and len(arr) == 2:
            return self._calculate_min_distance(arr)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            return np.array([self._calculate_min_distance(c) for c in arr], dtype=int)
        elif isinstance(target_coord, (list, tuple)):
            if len(target_coord) > 0 and isinstance(target_coord[0], (tuple, list, np.ndarray)):
                return np.array([self._calculate_min_distance(c) for c in target_coord], dtype=int)
            return self._calculate_min_distance(target_coord)
        return self._calculate_min_distance(target_coord)

    def _to_epoch_array(self, t):
        """Always returns a 1D float64 epoch-seconds array."""
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

    def _is_coord_arg(self, val):
        if isinstance(val, (tuple, list, np.ndarray)):
            arr = np.asarray(val)
            if (arr.ndim == 1 and len(arr) == 2) or (arr.ndim == 2 and arr.shape[1] == 2):
                return True
        return False

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and not isinstance(key[0], (float, int)):
            x, y = key
        else:
            x, y = key, None

        if y is not None:
            if self._is_coord_arg(x):
                coord_val, time_val = x, y
            elif self._is_coord_arg(y):
                coord_val, time_val = y, x
            else:
                coord_val, time_val = x, y

            coord_idx = self._get_closest_coord_idx(coord_val)
            times = self._to_epoch_array(time_val)
            n, weight = self._interp_indices_and_weights(times)

            coord_is_array = np.ndim(coord_idx) > 0
            time_is_array = len(n) > 1

            coord_results = {}
            for metric in self._metric_keys:
                m_data = self._metrics[metric]
                
                # Pointwise evaluation: ith coord at ith time
                if coord_is_array and time_is_array and len(coord_idx) == len(n):
                    arr_n = m_data[coord_idx, n]
                    arr_n1 = m_data[coord_idx, n + 1]
                # N coords at 1 time
                elif coord_is_array and not time_is_array:
                    arr_n = m_data[coord_idx, n[0]]
                    arr_n1 = m_data[coord_idx, n[0] + 1]
                # 1 coord at M times
                elif not coord_is_array and time_is_array:
                    arr_n = m_data[coord_idx, n]
                    arr_n1 = m_data[coord_idx, n + 1]
                # 1 coord at 1 time
                elif not coord_is_array and not time_is_array:
                    arr_n = m_data[coord_idx, n[0]]
                    arr_n1 = m_data[coord_idx, n[0] + 1]
                # Grid / Outer Product (N coords at M times)
                else:
                    arr_n = m_data[coord_idx[:, None], n[None, :]]
                    arr_n1 = m_data[coord_idx[:, None], (n + 1)[None, :]]

                coord_results[metric] = arr_n + weight * (arr_n1 - arr_n)

            return SolarResultProxy(coord_results)

        # Coordinate narrowing
        if isinstance(x, tuple) or (hasattr(x, '__iter__') and isinstance(x[0], tuple)):
            idx = self._get_closest_coord_idx(x)
            closest_coord = self._coords[idx]
            if isinstance(closest_coord[0], (int, float)):
                return SolarIrradiance({closest_coord: self._data[closest_coord]}, time_key=self._time_key, interval=self._interval, radius=self._radius)
            return SolarIrradiance({coord: self._data[coord] for coord in closest_coord}, time_key=self._time_key, interval=self._interval, radius=self._radius)

        return self._lookup_vectorized(x)

    def __repr__(self):
        return str(self._data)