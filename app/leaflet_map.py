"""Leaflet (Folium) map for the geofence simulation page.

Shows OpenStreetMap streets, the planned route, hazard / safe zones, a
time-animated tourist marker (press ▶ to replay the GPS trace minute by
minute) and a pop-up marker for every alert raised by the system.

Offline use: Leaflet only draws the map; the street images (tiles) come from
the tile server. Pass `tiles_url` pointing at locally stored tiles, e.g. a
folder served as /tiles/{z}/{x}/{y}.png, to show the map without internet.
"""
from __future__ import annotations

import folium
from folium.plugins import Fullscreen, TimestampedGeoJson

from src.geofence import CircleZone, PolygonZone

RISK_COLOR = {"LOW": "green", "MODERATE": "orange", "HIGH": "darkred", "CRITICAL": "red"}


def build_map(fixes, route, zones, alerts, tiles_url: str | None = None, center=(12.978, 77.592), zoom=15):
    if tiles_url:
        m = folium.Map(location=center, zoom_start=zoom, tiles=tiles_url, attr="Offline tiles © OpenStreetMap")
    else:
        m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    Fullscreen().add_to(m)

    # geofences
    for z in zones:
        color = "#c62828" if z.hazard else "#2e7d32"
        label = f"{'⚠ Hazard' if z.hazard else '✓ Safe'} zone: {z.name}"
        if isinstance(z, CircleZone):
            folium.Circle([z.lat, z.lon], radius=z.radius_m, color=color, weight=2, fill=True,
                          fill_opacity=0.18, tooltip=label).add_to(m)
        elif isinstance(z, PolygonZone):
            folium.Polygon(z.vertices, color=color, weight=2, fill=True, fill_opacity=0.18, tooltip=label).add_to(m)

    # planned route and the full GPS trace
    folium.PolyLine(route, color="#1565c0", weight=6, opacity=0.8, tooltip="Planned route").add_to(m)
    if fixes:
        folium.PolyLine([(f.lat, f.lon) for f in fixes], color="#424242", weight=2, dash_array="4 6",
                        opacity=0.7, tooltip="GPS trace").add_to(m)
        folium.Marker([fixes[0].lat, fixes[0].lon], tooltip="Start",
                      icon=folium.Icon(color="blue", icon="play", prefix="fa")).add_to(m)

        # time-animated tourist position (Leaflet TimeDimension); press play to replay
        features = [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [f.lon, f.lat]},
            "properties": {"time": f.t.isoformat(), "popup": f.t.strftime("%H:%M"),
                           "icon": "circle",
                           "iconstyle": {"fillColor": "#1565c0", "fillOpacity": 0.9, "stroke": True,
                                         "color": "#ffffff", "weight": 2, "radius": 8}},
        } for f in fixes]
        TimestampedGeoJson({"type": "FeatureCollection", "features": features}, period="PT1M",
                           duration="PT1M", add_last_point=True, auto_play=False, loop=False,
                           max_speed=20, loop_button=True, time_slider_drag_update=True,
                           date_options="HH:mm").add_to(m)

    # alerts
    for a in alerts:
        html = (f"<b>{a.final_class}</b> at {a.time[11:16]}<br>"
                f"On-device: {a.local_class} ({a.local_confidence:.2f})<br>"
                f"Decided by: {a.decided_by}<br>"
                f"{a.features['displacement_m']:.0f} m off route, idle {a.features['idle_min']:.0f} min<br>"
                f"Zones: {', '.join(a.zones) or 'none'}<br><b>Action:</b> {a.action}")
        if a.chain_record:
            html += f"<br>Blockchain tx: {a.chain_record[:12]}…"
        folium.Marker([a.lat, a.lon], popup=folium.Popup(html, max_width=320), tooltip=f"ALERT: {a.final_class}",
                      icon=folium.Icon(color=RISK_COLOR[a.final_class], icon="exclamation-triangle",
                                       prefix="fa")).add_to(m)
    return m
