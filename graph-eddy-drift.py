#!/usr/bin/env python3
# Eddy Probe Drift Data Collection
#
# Copyright (C) 2024 Eric Callahan <arksine.code@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import annotations
import argparse
import asyncio
import pathlib
import time
import re
import json
import bisect
from threading import Thread
from collections import deque
import websockets
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from numpy.polynomial import Polynomial
from dash import Dash, dcc, html, Input, Output, callback
from typing import Dict, Any, Optional, Tuple, List, Callable, Awaitable

plot_height = False
app = Dash("Eddy Probe Drift Analyzer")
app.layout = html.Div([
    html.H4("Probe Frequency Drift"),
    dcc.Graph(id="probe-drift-graph"),
    dcc.Interval(
        id="graph-interval",
        interval=1*15000,
        n_intervals=0
    )
])

@callback(
    Output("probe-drift-graph", "figure"),
    Input("graph-interval", "n_intervals")
)
def update_drift_graph(n):
    pdata = list(KlippyBridgeConnection.probe_data)
    temp = [p[0] for p in pdata]
    if plot_height:
        height = [round(p[2], 6) for p in pdata]
        stock_height = [round(p[3], 6) for p in pdata]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=temp, y=height, mode="markers", name="corrected height"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=temp, y=stock_height, mode="markers", name="stock height"
            )
        )
    else:
        df = pd.DataFrame({
            "temperature": temp,
            "frequency": [p[1] for p in pdata]
        })
        fig = px.scatter(df, x="temperature", y="frequency")
    return fig


INFO_REQUEST = "info"
SUB_REQUEST = "objects/subscribe"
QUERY_REQUEST = "objects/query"
DUMP_PROBE_REQUEST = "ldc1612/dump_ldc1612"
DUMP_BEACON_REQUEST = "beacon/dump"
SENSOR_NAME = "btt_eddy"
TEMP_SENSOR_NAME = "temperature_probe btt_eddy"
REQ_UPDATE_TIME = 60.
class KlippyBridgeConnection:
    probe_data: deque[tuple[float, ...]] = deque(maxlen=10000)

    def __init__(self, args: argparse.Namespace) -> None:
        url_match = re.match(r"((?:https?)|(?:wss?))://(.+)", args.url.lower())
        if url_match is None:
            raise Exception(f"Invalid Moonraker url {args.url}")
        scheme = url_match.group(1)
        host = url_match.group(2).rstrip("/")
        scheme = scheme.replace("http", "ws")
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.ldc_data_queue: deque[List[float]]  = deque(maxlen=100)
        self.last_temp: Tuple[float, float] = (0, 0)
        self.bridge_url = f"{scheme}://{host}/klippysocket"
        self.sensor_name = args.eddy_sensor
        self.has_beacon = self.sensor_name == "beacon"
        self.temp_sensor_name = f"temperature_probe {self.sensor_name}"
        self.need_collect_ldc = False
        self.stock_cal_freqs: List[float] = []
        self.stock_cal_zpos: List[float] = []
        self.conn_task: Optional[asyncio.Task] = None

    def load_ldc1612_calibration(self, cal):
        cal = sorted([(c[1], c[0]) for c in cal])
        self.stock_cal_freqs = [c[0] for c in cal]
        self.stock_cal_zpos = [c[1] for c in cal]

    def height_from_freq(self, freq):
        if not self.stock_cal_freqs or not self.stock_cal_zpos:
            return 99.0
        pos = bisect.bisect(self.stock_cal_freqs, freq)
        if pos >= len(self.stock_cal_zpos):
            zpos = -99.9
        elif pos == 0:
            zpos = 99.9
        else:
            this_freq = self.stock_cal_freqs[pos]
            prev_freq = self.stock_cal_freqs[pos - 1]
            this_zpos = self.stock_cal_zpos[pos]
            prev_zpos = self.stock_cal_zpos[pos - 1]
            gain = (this_zpos - prev_zpos) / (this_freq - prev_freq)
            offset = prev_zpos - prev_freq * gain
            zpos = freq * gain + offset
        return zpos

    async def start_connection(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.conn_task = self._loop.create_task(self._do_connect())
        await self.conn_task

    def stop(self) -> None:
        if self.conn_task is None:
            return
        self._loop.call_soon_threadsafe(self.conn_task.cancel)

    async def _do_connect(self) -> None:
        async with websockets.connect(self.bridge_url) as ws:
            asyncio.create_task(self._init_connection(ws))
            async for msg in ws:
                try:
                    resp: Dict[str, Any] = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                self._process_response(resp)

    async def _init_connection(self, ws: websockets.WebSocketClientProtocol):
        # Identify Connection
        info_req, fut = self._build_request(
            INFO_REQUEST, {"client_info": {"program": "Eddy Probe Analyzer"}}
        )
        await ws.send(info_req)
        await fut
        if self.has_beacon:
            print("Requesting Beacon Dump Endpoint...")
            self.need_collect_ldc = False
            dump_req, fut = self._build_request(
                DUMP_BEACON_REQUEST,
                {"response_template": {"method": "beacon_update"}}
            )
            await ws.send(dump_req)
            await fut
        else:
            print("Loading stock ldc1612 Calibration...")
            query_req, fut = self._build_request(
                QUERY_REQUEST, {"objects": {"configfile": ["settings"]}}
            )
            await ws.send(query_req)
            result: Dict[str, Any] = await fut
            cfg_status: Dict[str, Any] = result.get("status", {}).get("configfile", {})
            config: Dict[str, Any] = cfg_status.get("settings", {})
            probe_cfg: Dict[str, Any]
            probe_cfg = config.get(f"probe_eddy_current {self.sensor_name}", {})
            probe_cal: Optional[str] = probe_cfg.get("calibrate")
            if probe_cal is not None:
                cal = [
                    list(map(float, d.strip().split(':', 1)))
                    for d in probe_cal.split(',')
                ]
                self.load_ldc1612_calibration(cal)
                print("Successfully loaded ldc1612 calibration")
            else:
                print("Failed to load ldc1612 calibration")

            print(f"Subscribing to temperature updates for [{self.temp_sensor_name}]")
            temp_sub, fut = self._build_request(
                SUB_REQUEST,
                {
                    "objects": {self.temp_sensor_name: ["temperature"]},
                    "response_template": {"method": "status_update"}
                }
            )
            await ws.send(temp_sub)
            result = await fut
            eventtime = result["eventtime"]
            temp = result["status"][self.temp_sensor_name]["temperature"]
            self.last_temp = (temp, eventtime)
            self.need_collect_ldc = True
            dump_req, fut = self._build_request(
                DUMP_PROBE_REQUEST,
                {
                    "sensor": self.sensor_name,
                    "response_template": {"method": "ldc1612_update"}
                }
            )
            await ws.send(dump_req)
            await fut

    def _process_response(self, response: Dict[str, Any]) -> None:
        if "method" in response:
            method = response["method"]
            params = response.get("params", {})
            if isinstance(params, list):
                params = {"data": params}
            asyncio.create_task(self._execute_method(method, **params))
            return
        req_id: Optional[int] = response.get("id")
        if req_id is None:
            self.print_async(f"Invalid Response Received From Klippy: {response}")
            return
        fut = self.pending_requests.pop(req_id, None)
        if fut is None:
            self.print_async(f"No pending request matching response id: {req_id}")
            return
        if "result" in response:
            fut.set_result(response["result"])
        else:
            err = response.get("error", "Invalid Klippy Response")
            if isinstance(err, dict):
                err = err.get("message", "Invalid Klippy Response")
            fut.set_exception(Exception(err))

    def _build_request(
        self, method: str, params: Dict[str, Any] = {}
    ) -> Tuple[str, asyncio.Future]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        req: Dict[str, Any] = {"method": method}
        if params:
            req["params"] = params
        req["id"] = id(fut)
        self.pending_requests[id(fut)] = fut
        return json.dumps(req), fut

    async def _execute_method(self, method: str, **kwargs):
        func: Optional[Callable[[], Optional[Awaitable]]]
        func = getattr(self, f"_process_{method}", None)
        if func is None:
            return
        ret = func(**kwargs)
        if ret is not None:
            await ret

    def _process_status_update(
        self, eventtime: float, status: Dict[str, Dict[str, Any]]
    ) -> None:
        temp: Optional[float] = status.get(TEMP_SENSOR_NAME, {}).get("temperature")
        if temp is None:
            return
        last_temp, last_evt = self.last_temp
        if temp >= last_temp + 1. or eventtime > last_evt + REQ_UPDATE_TIME:
            self.print_async(f"Temp update: Last temp: {last_temp}, New Temp: {temp}")
            self.last_temp = (temp, eventtime)
            self.need_collect_ldc = True

    def _process_ldc1612_update(
        self, data: List[List[float]], errors: int, overflows: int
    ) -> None:
        # data comes in the format [eventtime, freq, calculated_z]
        if self.need_collect_ldc:
            self.ldc_data_queue.extend(data)
            if len(self.ldc_data_queue) >= 50:
                self._collect_samples()

    def _collect_samples(self) -> None:
        self.need_collect_ldc = False
        temp, evttime = self.last_temp
        ldc_queue = self.ldc_data_queue
        if self.has_beacon:
            # Beacon data comes in the format of [dist, temp, pos, freq, vel, eventtime]
            avg_freq = sum([d[3] for d in ldc_queue]) / len(ldc_queue)
            z_vals = [d[0] for d in ldc_queue if d[0] is not None]
            if z_vals:
                avg_z = sum(z_vals) / len(ldc_queue)
            else:
                avg_z = 99.
            stock_z = avg_z
            if len(ldc_queue[0]) == 7:
                raw_z_vals = [d[6] for d in ldc_queue if d[6] is not None]
                if raw_z_vals:
                    stock_z = sum(raw_z_vals) / len(raw_z_vals)
            self.print_async(
                "Beacon Frequency Data Updated\n"
                f"Temp: {temp:.2f} Freq: {avg_freq:.8f}, Z: {avg_z:.6f} "
                f"Stock Z: {stock_z:.6f}"
            )
        else:
            # LDC data comes in the format of [eventtime, freq, calculated_z]
            avg_freq = sum([d[1] for d in ldc_queue]) / len(ldc_queue)
            avg_z = sum([d[2] for d in ldc_queue]) / len(ldc_queue)
            stock_z = self.height_from_freq(avg_freq)
            self.print_async(
                "Eddy Frequency Data Updated\n"
                f"Temp: {temp:.2f} Freq: {avg_freq:.8f}, Z: {avg_z:.6f} "
                f"Stock Z: {stock_z:.6f}"
            )
        self.probe_data.append((temp, avg_freq, avg_z, stock_z))
        self.ldc_data_queue.clear()

    def _process_beacon_update(self, data: List[List[float]]):
        if self.need_collect_ldc:
            self.ldc_data_queue.extend(data)
            if len(self.ldc_data_queue) >= 50:
                self._collect_samples()
            return
        last_temp, last_evt = self.last_temp
        for i, sample in enumerate(data):
            temp = sample[1]
            evttime = sample[5]
            if temp >= last_temp + 1. or evttime > last_evt + REQ_UPDATE_TIME:
                self.print_async(
                    f"Temp update: Last temp: {last_temp}, New Temp: {temp}"
                )
                self.last_temp = (temp, evttime)
                self.need_collect_ldc = True
                self.ldc_data_queue.extend(data[i:])
                break

    def print_async(self, msg: str):
        self._loop.run_in_executor(None, print, msg)

def poly_to_str(poly: Polynomial):
    out = "y(x) = "
    deg = len(poly.coef) - 1
    for i, c in enumerate(reversed(poly.coef)):
        if int(c) == c:
            c == int(c)
        if c == 0:
            continue
        cur_deg = deg - i
        x_str = ""
        if cur_deg == 1:
            x_str = "x"
        elif cur_deg > 1:
            x_str = f"x^{cur_deg}"
        if i == 0:
            out = f"{out}{c}{x_str}"
        else:
            sym = "-" if c < 0 else "+"
            out = f"{out} {sym} {abs(c)}{x_str}"
    return out

def fit_poly():
    # x = temperatures, y = frequencies
    pdata = KlippyBridgeConnection.probe_data
    if not pdata:
        return
    base_freq = pdata[0][1]
    x = [p[0] for p in pdata]
    y = [p[1] - base_freq for p in pdata]
    if len(x) < 3:
        return [], "Failed to fit polynomial"
    try:
        poly = Polynomial.fit(x, y, 2).convert()
    except Exception:
        return [], "Failed to fit polynomial"
    # Convert to string
    poly_str = poly_to_str(poly)
    return poly.coef.tolist(), poly_str

def dump_samples(desc: Optional[str]):
    pdata = KlippyBridgeConnection.probe_data
    if not pdata:
        return
    coefs, poly_str = fit_poly()
    print(f"Fitted Polynomial: {poly_str}")
    pdict = {
        "polynomial": {
            "repr": poly_str,
            "coefficients": coefs
        },
        "legend": ["temperature", "frequency", "height", "stock_height"],
        "samples": list(pdata)
    }
    postfix = desc or time.strftime("%Y%m%d_%H%M%S")
    parent_dir = pathlib.Path(__file__).parent
    samples_path = parent_dir.joinpath(f"data-samples-{postfix}.json")
    samples_path.write_text(json.dumps(pdict, indent=2))

def start_klippy_connection(conn: KlippyBridgeConnection) -> None:
    try:
        asyncio.run(conn.start_connection())
    except asyncio.CancelledError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Eddy Probe Drift Analyzer")
    parser.add_argument(
        "-e", "--eddy-sensor", default=SENSOR_NAME,
        help="Name of the eddy current sensor config object"
    )
    parser.add_argument(
        "-a", "--dash-addr", default="127.0.0.1:8050",
        help="Address to bind dash server to"
    )
    parser.add_argument(
        "-d", "--desc", default=None,
        help="Description for output file"
    )
    parser.add_argument(
        "-z", "--plot-z", action="store_true",
        help="plot z height"
    )
    parser.add_argument(
        "url", metavar="<moonraker url>",
        help="URL to Moonraker instance"
    )
    args = parser.parse_args()
    if args.plot_z:
        global plot_height
        plot_height = True
    conn = KlippyBridgeConnection(args)
    conn_thread = Thread(target=start_klippy_connection, args=[conn])
    conn_thread.start()
    addr_parts = args.dash_addr.split(":", maxsplit=1)
    host = addr_parts[0]
    port = addr_parts[1] if len(addr_parts) == 2 else "8050"
    app.run(
        host=host, port=port, debug=True, use_reloader=False
    )
    conn.stop()
    conn_thread.join()
    dump_samples(args.desc)


if __name__ == "__main__":
    main()
