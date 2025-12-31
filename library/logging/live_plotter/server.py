import sys
import threading
import json
import argparse
import os
import logging
import uuid
import numpy as np

from flask import Flask, jsonify, send_from_directory
from werkzeug.serving import WSGIRequestHandler


class PlotterState:
    """
    Manages the state of the live plotter server.

    Stores timestep counts, schedule data, and settings data.
    """
    def __init__(self):
        self.timestep_counts = np.zeros(1000, dtype=np.int64)
        self.schedule_data = None
        self.settings_data = None
        self.session_id = str(uuid.uuid4())
        self.data_lock = threading.Lock()

    def reset(self):
        """Resets the plotter state."""
        with self.data_lock:
            self.timestep_counts.fill(0)
            self.schedule_data = None
            self.settings_data = None
            self.session_id = str(uuid.uuid4())
            print("Plotter: State has been reset.")

state = PlotterState()

def run_plotter_server(port: int):
    """
    Runs the live plotter server.

    Args:
        port: The port number to run the server on.
    """
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    app = Flask(__name__, static_folder=web_dir)

    WSGIRequestHandler.log_request = lambda *args, **kwargs: None
    app.logger.disabled = True
    logging.getLogger('werkzeug').disabled = True

    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'dashboard.html')

    @app.route('/settings_data')
    def get_settings_data():
        with state.data_lock:
            if state.settings_data is None:
                return jsonify({"status": "not_ready", "session_id": state.session_id})
            return jsonify({"status": "ready", "data": state.settings_data, "session_id": state.session_id})

    @app.route('/schedule_data')
    def get_schedule_data():
        with state.data_lock:
            if state.schedule_data is None:
                return jsonify({"status": "not_ready"})
            return jsonify({"status": "ready", "data": state.schedule_data})

    @app.route('/distribution_data')
    def get_distribution_data():
        with state.data_lock:
            return jsonify({"status": "ready", "data": state.timestep_counts.tolist(), "session_id": state.session_id})

    def data_listener():
        for line in sys.stdin:
            line = line.strip()
            if not line: continue

            if line == "RESET::":
                state.reset()
                print("Plotter: Received RESET command. Data cleared and new session started.")
                continue

            if line.startswith("SETTINGS::"):
                try:
                    with state.data_lock:
                        state.settings_data = json.loads(line[len("SETTINGS::"):])
                    print("Plotter: Received settings data.")
                except Exception as e:
                    print(f"Plotter: Error processing settings data: {e}")
                continue

            if line.startswith("SCHEDULE::"):
                try:
                    alphas_cumprod = np.fromstring(line[len("SCHEDULE::"):], sep=',')
                    signal_factor = np.sqrt(alphas_cumprod)
                    noise_factor = np.sqrt(1 - alphas_cumprod)
                    with state.data_lock:
                        state.schedule_data = {
                            "labels": list(range(len(alphas_cumprod))),
                            "signalData": signal_factor.tolist(),
                            "noiseData": noise_factor.tolist(),
                        }
                    print("Plotter: Received noise schedule data.")
                except Exception as e:
                    print(f"Plotter: Error processing schedule data: {e}")
                continue

            try:
                timesteps = np.fromstring(line, dtype=np.int64, sep=',')
                with state.data_lock:
                    unique, counts = np.unique(timesteps, return_counts=True)
                    valid_indices = unique < len(state.timestep_counts)
                    state.timestep_counts[unique[valid_indices]] += counts[valid_indices]
            except (ValueError, IndexError):
                continue

    listener_thread = threading.Thread(target=data_listener, daemon=True)
    listener_thread.start()

    print(f"\n\n ✨ Interactive Dashboard is running at: http://localhost:{port} ✨ \n\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Timestep Plotter Server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    run_plotter_server(args.port)
