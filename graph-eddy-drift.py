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
from threading import Thread
from collections import deque
import websockets
import plotly.express as px
import pandas as pd
from numpy.polynomial import Polynomial
from dash import Dash, dcc, html, Input, Output, callback
from typing import Dict, Any, Optional, Tuple, List, Callable, Awaitable

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
    df = pd.DataFrame({
        "temperature": [p[0] for p in pdata],
        "frequency": [p[1] for p in pdata]
    })
    fig = px.scatter(df, x="temperature", y="frequency")
    return fig


INFO_REQUEST = "info"
SUB_REQUEST = "objects/subscribe"
DUMP_PROBE_REQUEST = "ldc1612/dump_ldc1612"
SENSOR_NAME = "btt_eddy"
TEMP_SENSOR_NAME = "probe_drift"
REQ_UPDATE_TIME = 60.

# TODO:
# Plot (use plotly?  Use Dash?)

class KlippyBridgeConnection:
    probe_data = deque(maxlen=10000)

    def __init__(self, args: argparse.Namespace) -> None:
        url_match = re.match(r"((?:https?)|(?:wss?))://(.+)", args.url.lower())
        if url_match is None:
            raise Exception(f"Invalid Moonraker url {args.url}")
        scheme = url_match.group(1)
        host = url_match.group(2).rstrip("/")
        scheme = scheme.replace("http", "ws")
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.ldc_data_queue = deque(maxlen=100)
        self.last_temp: Tuple[float, float] = (0, 0)
        self.bridge_url = f"{scheme}://{host}/klippysocket"
        self.sensor_name = args.eddy_sensor
        self.temp_sensor_name = TEMP_SENSOR_NAME
        if args.temp_sensor is not None:
            self.temp_sensor_name = f"temperature_sensor {args.temp_sensor}"
        self.need_collect_ldc = False
        self.conn_task: Optional[asyncio.Task] = None

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
        print(f"Subscribing to temperature updates for [{self.temp_sensor_name}]")
        temp_sub, fut = self._build_request(
            SUB_REQUEST,
            {
                "objects": {self.temp_sensor_name: ["temperature"]},
                "response_template": {"method": "status_update"}
            }
        )
        await ws.send(temp_sub)
        result: Dict[str, Any] = await fut
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
        req = {"method": method}
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
        # LDC data comes in the format of [eventtime, freq, calculated_z]
        avg_freq = sum([d[1] for d in ldc_queue]) / len(ldc_queue)
        avg_z = sum([d[2] for d in ldc_queue]) / len(ldc_queue)
        self.probe_data.append((temp, avg_freq))
        self.ldc_data_queue.clear()
        self.print_async(
            "Eddy Frequency Data Updated\n"
            f"Temp: {temp:.2f} Freq: {avg_freq:.8f}, Z: {avg_z:.6f}"
        )

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
        poly = Polynomial.fit(x, y, 1).convert()
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
        "legend": ["temperature", "frequency"],
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
        "-t", "--temp-sensor", default=None,
        help="Name of the temperature sensor config object"
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
        "url", metavar="<moonraker url>",
        help="URL to Moonraker instance"
    )
    args = parser.parse_args()
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