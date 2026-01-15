import importlib.util
import sys
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

def getOutputName():
    outputName = "output.csv"
    return outputName

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
from filterpy.kalman import (
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
    unscented_transform,
)

"""
vehicle less than a 1 meter
frequency update every 1 second
plot the predictions on the map

state:
    lat: latitude in degrees [-90, 90]
    lon: longitude in degrees [-180, 180]
    v_east: velocity east in m/s
    v_north: velocity north in m/s
    a_east: acceleration east in m/s^2 (not measured directly)
    a_north: acceleration north in m/s^2 (not measured directly)
    
for dt:
    store last measurement time
"""


def normalize_longitude(lat: float) -> float:
    # normalize angle to [0, 360)
    lat %= 360
    if lat > 180:
        # shift to (-180, 180]
        lat -= 360
    return lat


def normalize_latitude(lon: float) -> float:
    # normalize angle to [0, 180)
    lon %= 180
    if lon > 90:
        # shift to (-90, 90]
        lon -= 180
    return lon


@dataclass(frozen=True)
class StateEstimate:
    # latitude in degrees [-90, 90]
    latitude: float
    latitude_std_dev: float

    # longitude in degrees [-180, 180]
    longitude: float
    longitude_std_dev: float

    timestamp_utc: datetime


class CsvMeasurement:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        speed_knots: float,
        course_deg: float,
        timestamp_utc: datetime,
    ) -> None:
        speed_ms = speed_knots * 0.514444
        if speed_ms < 0:
            speed_ms = 0

        if course_deg < 0:
            course_deg = 0

        if speed_ms != 0:
            v_east = speed_ms * np.sin(np.radians(course_deg))
            v_north = speed_ms * np.cos(np.radians(course_deg))
        else:
            v_east = 0
            v_north = 0

        # latitude in degrees [-90, 90]
        self.latitude: float = latitude

        # longitude in degrees [-180, 180]
        self.longitude: float = longitude

        # velocity east in m/s
        self.v_east: float = v_east

        # velocity north in m/s
        self.v_north: float = v_north

        # datetime in UTC
        self.timestamp_utc: datetime = timestamp_utc

    def to_array(self) -> np.ndarray:
        """Convert the measurement to a numpy array for the Kalman filter update input"""
        return np.array([self.latitude, self.longitude, self.v_east, self.v_north])

    def __str__(self) -> str:
        return f"CsvMeasurement(latitude={self.latitude}, longitude={self.longitude}, v_east={self.v_east}, v_north={self.v_north}, timestamp_utc={self.timestamp_utc})"


class Measurement:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        speed_ms: float,
        course_deg: float,
        timestamp_utc: datetime,
    ) -> None:
        if speed_ms < 0:
            speed_ms = 0

        if course_deg < 0:
            course_deg = 0

        if speed_ms != 0:
            v_east = speed_ms * np.sin(np.radians(course_deg))
            v_north = speed_ms * np.cos(np.radians(course_deg))
        else:
            v_east = 0
            v_north = 0

        # latitude in degrees [-90, 90]
        self.latitude: float = latitude

        # longitude in degrees [-180, 180]
        self.longitude: float = longitude

        # velocity east in m/s
        self.v_east: float = v_east

        # velocity north in m/s
        self.v_north: float = v_north

        # datetime in UTC
        self.timestamp_utc: datetime = timestamp_utc

    def to_array(self) -> np.ndarray:
        """Convert the measurement to a numpy array for the Kalman filter update input"""
        return np.array([self.latitude, self.longitude, self.v_east, self.v_north])

    def __str__(self) -> str:
        return f"Measurement(latitude={self.latitude}, longitude={self.longitude}, v_east={self.v_east}, v_north={self.v_north}, timestamp_utc={self.timestamp_utc})"


class ExportedKalmanFilter: ...


class KalmanFilter:
    # default time step in seconds
    _DEFAULT_DT = 1.0

    @staticmethod
    def _create_kalman_filter(
        initial_latitude_deg: float = 0,
        initial_longitude_deg: float = 0,
    ) -> UnscentedKalmanFilter:
        # sigma points to parameterize the distribution
        points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2.0, kappa=0.0)

        ukf = UnscentedKalmanFilter(
            dim_x=6,
            dim_z=4,
            dt=KalmanFilter._DEFAULT_DT,
            fx=KalmanFilter._fx,
            hx=KalmanFilter._hx,
            points=points,
        )

        # initial state
        ukf.x = np.array(
            [
                initial_latitude_deg,  # latitude in degrees [-90, 90]
                initial_longitude_deg,  # longitude in degrees [-180, 180]
                0,  # velocity east in m/s
                0,  # velocity north in m/s
                0,  # acceleration east in m/s^2 (not measured directly)
                0,  # acceleration north in m/s^2 (not measured directly)
            ]
        )

        # initial state estimate uncertainty (with variance)
        # 1 degree latitude is approximately 111 km
        # 1 m/s is 3.6 km/h
        ukf.P = np.diag(
            [
                0.002**2,  # 200 meters
                0.002**2,  # 200 meters
                5.0**2,  # 18 km/h
                5.0**2,  # 18 km/h
                1.5**2,  # 5.4 km/h/s
                1.5**2,  # 5.4 km/h/s
            ]
        )

        # measurement uncertainty (with variance)
        ukf.R = np.diag(
            [
                0.0001**2,  # 10 meters (typical open sky)
                0.0001**2,  # 10 meters (typical open sky)
                1.0**2,  # 3.6 km/h
                1.0**2,  # 3.6 km/h
            ]
        )

        # process (model) uncertainty (with variance)
        ukf.Q = np.diag(
            [
                0.0001**2,  # 10 meter
                0.0001**2,  # 10 meter
                1.0**2,  # 3.6 km/h
                1.0**2,  # 3.6 km/h
                1.0**2,  # 3.6 km/h/s
                1.0**2,  # 3.6 km/h/s
            ]
        )

        return ukf

    @staticmethod
    def _fx(x: np.ndarray, dt: float) -> np.ndarray:
        """State transition function"""
        lat, lon, v_east, v_north, a_east, a_north = x

        # earth radius in meters
        R = 6370100

        angular_v_lat_rad = v_north / R
        angular_a_lat_rad = a_north / R
        lat_delta_rad = angular_v_lat_rad * dt + 0.5 * angular_a_lat_rad * dt**2
        lat_delta_deg = np.degrees(lat_delta_rad)

        angular_v_lon_rad = v_east / (R * np.cos(np.radians(lat)))
        angular_a_lon_rad = a_east / (R * np.cos(np.radians(lat)))
        lon_delta_rad = angular_v_lon_rad * dt + 0.5 * angular_a_lon_rad * dt**2
        lon_delta_deg = np.degrees(lon_delta_rad)

        lat_new = normalize_latitude(lat + lat_delta_deg)
        lon_new = normalize_longitude(lon + lon_delta_deg)

        # update velocity components
        new_v_east = v_east + a_east * dt
        new_v_north = v_north + a_north * dt

        return np.array([lat_new, lon_new, new_v_east, new_v_north, a_east, a_north])

    @staticmethod
    def _hx(x: np.ndarray) -> np.ndarray:
        """Measurement function"""
        return np.array([x[0], x[1], x[2], x[3]])

    @classmethod
    def create(
        cls, initial_latitude_deg: float, initial_longitude_deg: float
    ) -> "KalmanFilter":
        return cls(
            kalman_filter=cls._create_kalman_filter(
                initial_latitude_deg=initial_latitude_deg,
                initial_longitude_deg=initial_longitude_deg,
            ),
            last_prediction_time=None,
        )

    @classmethod
    def load(cls, exported_filter: ExportedKalmanFilter) -> "KalmanFilter": ...

    def __init__(
        self,
        kalman_filter: UnscentedKalmanFilter,
        last_prediction_time: datetime | None,
    ) -> None:
        self._filter: UnscentedKalmanFilter = kalman_filter
        self._last_update_time: datetime | None = last_prediction_time

    def update(self, measurement: Measurement) -> StateEstimate:
        if self._last_update_time is None:
            dt = KalmanFilter._DEFAULT_DT
            self._last_update_time = measurement.timestamp_utc
        else:
            dt = (measurement.timestamp_utc - self._last_update_time).total_seconds()
            if dt <= 0:
                raise ValueError(
                    f"Timestamps must be strictly increasing,\nmeasurement: {measurement.timestamp_utc}\nlast update: {self._last_update_time}"
                )
            self._last_update_time = measurement.timestamp_utc
        self._filter.predict(dt=dt)
        self._filter.update(measurement.to_array())
        return StateEstimate(
            latitude=float(self._filter.x[0]),
            latitude_std_dev=float(np.sqrt(self._filter.P[0, 0])),
            longitude=float(self._filter.x[1]),
            longitude_std_dev=float(np.sqrt(self._filter.P[1, 1])),
            timestamp_utc=measurement.timestamp_utc,
        )

    def predict(self, dt: float) -> StateEstimate:
        """Does not modify the internal state of the filter"""
        sigmas = self._filter.points_fn.sigma_points(self._filter.x, self._filter.P)
        sigmas_f = self._filter.sigmas_f.copy()
        for i, s in enumerate(sigmas):
            sigmas_f[i] = self._filter.fx(s, dt)
        x, P = unscented_transform(
            sigmas_f,
            self._filter.Wm,
            self._filter.Wc,
            self._filter.Q,
            self._filter.x_mean,
            self._filter.residual_x,
        )
        assert self._last_update_time is not None
        timestamp = self._last_update_time + timedelta(seconds=dt)
        return StateEstimate(
            latitude=float(x[0]),
            latitude_std_dev=float(np.sqrt(P[0, 0])),
            longitude=float(x[1]),
            longitude_std_dev=float(np.sqrt(P[1, 1])),
            timestamp_utc=timestamp,
        )

    def export(self) -> ExportedKalmanFilter: ...


def load_csv_data() -> List[Measurement]:
    import pandas as pd

    data_path = "function/embedded_files/output.csv"
    df = pd.read_csv(data_path)

    # convert string lat (i.e. 37.996349 N) and lon (i.e. 23.762732 E) to decimal degrees
    df["latitude"] = df["latitude"].apply(lambda x: float(x.split(" ")[0]))
    df["longitude"] = df["longitude"].apply(lambda x: float(x.split(" ")[0]))

    # convert string speed
    df["speed"] = df["speed"].apply(lambda x: float(x.split(" ")[0]))

    # convert string course
    df["course"] = df["course"].apply(lambda x: float(x.split(" ")[0]))

    # concat date and time columns
    df["date_str"] = df["date"].apply(lambda x: str(x))
    df["utc_time_str"] = df["utc_time"].apply(lambda x: str(x))
    df["timestamp"] = df["date_str"] + df["utc_time_str"]

    # transform each row to a list of measurements
    measurements = df[
        ["latitude", "longitude", "speed", "course", "timestamp"]
    ].values.tolist()
    for i in range(len(measurements)):
        timestamp = datetime.strptime(measurements[i][4], "%d%m%y%H%M%S.%f")
        timestamp = timestamp.replace(tzinfo=timezone.utc)
        measurements[i] = CsvMeasurement(
            latitude=measurements[i][0],
            longitude=measurements[i][1],
            speed_knots=measurements[i][2],
            course_deg=measurements[i][3],
            timestamp_utc=timestamp,
        )

    return measurements


def append_to_csv(file_path: str, new_row: dict, max_rows: int) -> None:
    fieldnames = new_row.keys()
    try:
        with open(file_path, "r", newline="") as f:
            row_count = sum(1 for _ in csv.DictReader(f, fieldnames=fieldnames))
    except FileNotFoundError:
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        row_count = 0
    excess = row_count + 1 - max_rows
    if excess > 0:
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            rows = [row for row in reader]
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[excess:])
            writer.writerow(new_row)
    else:
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(new_row)


kf: KalmanFilter | None = None


def handle(req):
    global kf

    if not req:
        last_update_utc = None
        if kf is not None:
            last_update_utc = kf._last_update_time.isoformat()
        return {
            "status": "ok",
            "filter_last_update_utc": last_update_utc,
        }

    # latitude_deg
    # longitude_deg
    # speed_ms
    # course_deg
    # timestamp_utc_iso_8601
    # prediction_dt_seconds
    req = json.loads(req)
    measurement = Measurement(
        latitude=req["latitude_deg"],
        longitude=req["longitude_deg"],
        speed_ms=req["speed_ms"],
        course_deg=req["course_deg"],
        timestamp_utc=datetime.fromisoformat(req["timestamp_utc_iso_8601"]),
    )
    prediction_dt = float(req["prediction_dt_seconds"])

    if kf is None:
        kf = KalmanFilter.create(
            initial_latitude_deg=measurement.latitude,
            initial_longitude_deg=measurement.longitude,
        )

    try:
        current_state_estimate = kf.update(measurement)
        predicted_next_state = kf.predict(dt=prediction_dt)
    except Exception as e:
        kf = None
        return {
            "status": "error (filter restarted)",
            "message": str(e),
        }

    append_to_csv(
        file_path="function/embedded_files/output.csv",
        new_row={
            "lat_m": measurement.latitude,
            "lon_m": measurement.longitude,
            "timestamp_m": measurement.timestamp_utc.isoformat(),
            "lat_curr": current_state_estimate.latitude,
            "lat_std_dev_curr": current_state_estimate.latitude_std_dev,
            "lon_curr": current_state_estimate.longitude,
            "lon_std_dev_curr": current_state_estimate.longitude_std_dev,
            "timestamp_curr": current_state_estimate.timestamp_utc.isoformat(),
            "lat_next": predicted_next_state.latitude,
            "lat_std_dev_next": predicted_next_state.latitude_std_dev,
            "lon_next": predicted_next_state.longitude,
            "lon_std_dev_next": predicted_next_state.longitude_std_dev,
            "timestamp_next": predicted_next_state.timestamp_utc.isoformat(),
        },
        max_rows=50,
    )

    return {
        "measurement": {
            "latitude": measurement.latitude,
            "longitude": measurement.longitude,
            "timestamp_utc": measurement.timestamp_utc.isoformat(),
        },
        "current_state_estimate": {
            "latitude": current_state_estimate.latitude,
            "latitude_std_dev": current_state_estimate.latitude_std_dev,
            "longitude": current_state_estimate.longitude,
            "longitude_std_dev": current_state_estimate.longitude_std_dev,
            "timestamp_utc": current_state_estimate.timestamp_utc.isoformat(),
        },
        "predicted_next_state": {
            "latitude": predicted_next_state.latitude,
            "latitude_std_dev": predicted_next_state.latitude_std_dev,
            "longitude": predicted_next_state.longitude,
            "longitude_std_dev": predicted_next_state.longitude_std_dev,
            "timestamp_utc": predicted_next_state.timestamp_utc.isoformat(),
        },
    }
