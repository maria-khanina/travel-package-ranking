"""Departure countries and hubs of travelers."""

from src.settings.records import HubRecord, OriginRecord


ORIGINS: dict[str, OriginRecord] = {
    "US": OriginRecord(
        weight=0.45,
        budget=1.0,
        alpha=-0.05,
        hubs=[
            HubRecord(name="JFK", latitude=40.64, longitude=-73.78, weight=0.4),
            HubRecord(name="ORD", latitude=41.98, longitude=-87.9, weight=0.25),
            HubRecord(name="LAX", latitude=33.94, longitude=-118.41, weight=0.25),
            HubRecord(name="MIA", latitude=25.79, longitude=-80.29, weight=0.1),
        ],
    ),
    "UK": OriginRecord(
        weight=0.22,
        budget=0.85,
        alpha=0.1,
        hubs=[
            HubRecord(name="LHR", latitude=51.47, longitude=-0.45, weight=0.6),
            HubRecord(name="MAN", latitude=53.35, longitude=-2.28, weight=0.4),
        ],
    ),
    "CA": OriginRecord(
        weight=0.12,
        budget=0.9,
        alpha=0.0,
        hubs=[
            HubRecord(name="YYZ", latitude=43.68, longitude=-79.63, weight=0.65),
            HubRecord(name="YVR", latitude=49.19, longitude=-123.18, weight=0.35),
        ],
    ),
    "AU": OriginRecord(
        weight=0.12,
        budget=0.95,
        alpha=-0.05,
        hubs=[
            HubRecord(name="SYD", latitude=-33.94, longitude=151.18, weight=0.55),
            HubRecord(name="MEL", latitude=-37.67, longitude=144.84, weight=0.45),
        ],
    ),
    "IE": OriginRecord(
        weight=0.05,
        budget=0.85,
        alpha=0.08,
        hubs=[HubRecord(name="DUB", latitude=53.43, longitude=-6.27, weight=1.0)],
    ),
    "NZ": OriginRecord(
        weight=0.04,
        budget=0.8,
        alpha=0.1,
        hubs=[HubRecord(name="AKL", latitude=-37.01, longitude=174.79, weight=1.0)],
    ),
}
