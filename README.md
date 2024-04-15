## Eddy Probe Temperature Analysis

This repository contains two scripts that collect and plot data received from
dc1612 "Eddy Current Probes" running Klipper.  Klipper and Moonraker are required
to be installed to perform the data collection.  As of this writing users should
be working off of the
[probe drift development branch](https://github.com/Arksine/klipper/tree/dev-probe-drift-20240304)
to use these scripts.

The klipper configuration should include something like the following:

```
[mcu eddy]
serial: /dev/serial/by-id/usb-Klipper_rp2040_45503571290B1068-if00

[probe_eddy_current btt_eddy]
sensor_type: ldc1612
x_offset: 0.0
y_offset: 25.0
z_offset: .5
i2c_mcu: eddy
i2c_bus: i2c0f
speed: 7.5
lift_speed: 20

[temperature_probe btt_eddy]
sensor_type: Generic 3950
sensor_pin: eddy:gpio26

[temperature_sensor eddy_mcu]
sensor_type: temperature_mcu
sensor_mcu: eddy
gcode_id: EDY
min_temp: 0
max_temp: 120

```

The names assigned to the `probe_eddy_current` and `temperature_sensor` sections will be
required to run the data collection script.

### Setup

Clone this repo, create a virtualenv and install the requirements:

```
git clone https://github.com/Arksine/eddy-probe-temp.git
cd eddy-probe-temp
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

While the scripts in this repo are intended to run on a desktop class PC, they can be
installed and run on a Raspberry Pi.  The RPi may require libopenblas:

```
sudo apt update
sudo apt install libopenblas-dev
```

### graph-eddy-drift.py

```
./graph-eddy-drift.py -h
usage: graph-eddy-drift.py [-h] [-e EDDY_SENSOR] [-t TEMP_SENSOR] [-a DASH_ADDR] [-d DESC] <moonraker url>

Eddy Probe Drift Analyzer

positional arguments:
  <moonraker url>       URL to Moonraker instance

options:
  -h, --help            show this help message and exit
  -e EDDY_SENSOR, --eddy-sensor EDDY_SENSOR
                        Name of the eddy current sensor config object
  -a DASH_ADDR, --dash-addr DASH_ADDR
                        Address to bind dash server to
  -d DESC, --desc DESC  Description for output file
  -z, --plot-z          plot z height instead of frequency
```

The `graph-eddy-drift` script connects to Moonraker through a websocket, thus it is not
required to run this script on the host machine running Klipper.  Moonraker's url is a
required positional argument. After establishing a connection to Klipper through Moonraker
this script will launch a [Dash](https://dash.plotly.com/) based http server listening
`127.0.0.1:8050` by default.  This address may be specified using the `-a` option.  For
example, to listen to all ipv4 interfaces on port 8050 specify `-a 0.0.0.0:8050`. Navigating
to the dash host in a web browser will show a live plot of the drift as data is collected.
The interface can also be used to save a snapshot of the plot if desired.

The server may be closed by pressing Ctrl-C in the terminal window running the script.
At this time the collected data will be dumped to a json file in the repo.  The file
name will be in the format of `data-samples-{desc}.json`, where `desc` defaults to a
timestamp. The `-d` option can be used to customize the description.

If the `-z` option is provided the script will plot Z height against temperature instead
of frequency.  The eddy probe will need a valid calibration to plot an accurate height.
If the eddy probe has a drift calibration configured then the corrected height will
also be included in the plot.

When launching the script, the probe name should be specified using the `-e` option.
For example, assume we want to launch the script with the following conditions:

- `[probe_eddy_current btt_eddy]` is configured in Klipper
- `[temperature_sensor btt_eddy]` is configured in Klipper
- We want to bind the dash server to all ipv4 interfaces on 8050
- We want to add a `85c-2mm` description to the file name to note the bed temp
  and probe height we are recording.
- The Moonraker URL we want to connect to is located at `http://my-printer.local`

```
# activate the venv if necessary
source ./venv/bin/activate
./graph-eddy-drift.py -e btt_eddy -a 0.0.0.0:8050 -d 85c-2mm http://my-printer.local
```

If the script is run on the localhost go to `http://localhost:8050` to view the live plot.
Exit the script with `Ctrl-C`, which will stop collection and dump the collected data
to a `data-samples` file in the repo.

### graph-composite.py

```
./graph-composite.py -h
usage: graph-composite.py [-h] [-i INPUT_DIRECTORY] [-o OUTPUT]

Plot Composite Graph

options:
  -h, --help            show this help message and exit
  -i INPUT_DIRECTORY, --input-directory INPUT_DIRECTORY
                        Path to input directory
  -o OUTPUT, --output OUTPUT
                        Output file path
```

The `graph-composite` will generate a single, composite plot containing
multiple graphs generated from `data-sample-{desc}.json` files saved in
the repo directory. If `-o` is specified the graph will be saved to an
image file, otherwise matplotlib will launch a live preview.  The `-i`
option may be used to read `data-sample` files from a directory other
than the repo folder.

