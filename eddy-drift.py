#!/usr/bin/env python3
# Eddy Probe Drift Data Collection
#
# Copyright (C) 2024 Eric Callahan <arksine.code@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import annotations
import argparse
import asyncio
import re
import json
import logging
from collections import deque
import websockets
from typing import Dict, Any, Optional, Tuple, List, Callable, Awaitable

logging.basicConfig(level=logging.INFO)

INFO_REQUEST = "info"
SUB_REQUEST = "objects/subscribe"
DUMP_PROBE_REQUEST = "ldc1612/dump_ldc1612"
SENSOR_NAME = "btt_eddy"
TEMP_SENSOR_NAME = "temperature_sensor eddy_probe"
REQ_UPDATE_TIME = 30.

# TODO:
# Plot (use plotly?  Use Dash?)

class KlippyBridgeConnection:
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
        self.probe_data: List[Tuple[float, float, float]] = []
        self.bridge_url = f"{scheme}://{host}/klippysocket"
        self.need_collect_ldc = False

    async def start_connection(self) -> None:
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
        temp_sub, fut = self._build_request(
            SUB_REQUEST,
            {
                "objects": {TEMP_SENSOR_NAME: ["temperature"]},
                "response_template": {"method": "status_update"}
            }
        )
        await ws.send(temp_sub)
        result: Dict[str, Any] = await fut
        eventtime = result["eventtime"]
        temp = result["status"][TEMP_SENSOR_NAME]["temperature"]
        self.last_temp = (temp, eventtime)
        dump_req, fut = self._build_request(
            DUMP_PROBE_REQUEST,
            {
                "sensor": SENSOR_NAME,
                "response_template": {"method": "ldc1612_update"}
            }
        )
        await ws.send(dump_req)
        result: Dict[str, Any] = await fut
        self.need_collect_ldc = True

    def _process_response(self, response: Dict[str, Any]) -> None:
        if "method" in response:
            method = response["method"]
            params = response.get("params", {})
            asyncio.create_task(self._execute_method(method, **params))
            return
        req_id: Optional[int] = response.get("id")
        if req_id is None:
            logging.info(f"Invalid Response Received From Klippy: {response}")
            return
        fut = self.pending_requests.pop(req_id, None)
        if fut is None:
            logging.info(f"No pending request matching response id: {req_id}")
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
        if temp > last_temp + 1. or eventtime > last_evt + REQ_UPDATE_TIME:
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
        self.probe_data.append((temp, avg_freq, avg_z))
        self.ldc_data_queue.clear()
        logging.info(
            "Eddy Frequency Data Updated\n"
            f"Temp: {temp:.2f} Freq: {avg_freq:.8f}, Z: {avg_z:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Eddy Probe Drift Analyzer")
    parser.add_argument(
        "url", metavar="<moonraker url>",
        help="URL to Moonraker instance"
    )
    args = parser.parse_args()
    conn = KlippyBridgeConnection(args)
    try:
        asyncio.run(conn.start_connection())
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
