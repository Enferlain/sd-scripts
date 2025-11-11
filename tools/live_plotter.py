import sys
import threading
import json
import argparse
import os
import logging

import numpy as np
from flask import Flask, jsonify, send_from_directory
# --- REFINEMENT: Import for advanced log silencing ---
from werkzeug.serving import WSGIRequestHandler

def run_plotter_server(port):
    # In-memory "databases"
    timestep_counts = np.zeros(1000, dtype=np.int64)
    schedule_data = None
    settings_data = None
    data_lock = threading.Lock()

    # --- Flask App Setup ---
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    app = Flask(__name__, static_folder=web_dir)

    # --- REFINEMENT: Advanced log silencing ---
    # This is more effective than changing the logger level.
    WSGIRequestHandler.log_request = lambda *args, **kwargs: None
    app.logger.disabled = True
    logging.getLogger('werkzeug').disabled = True

    # --- API and File Serving Routes ---
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'dashboard.html')

    @app.route('/settings_data')
    def get_settings_data():
        with data_lock:
            if settings_data is None:
                return jsonify({"status": "not_ready"})
            return jsonify({"status": "ready", "data": settings_data})

    @app.route('/schedule_data')
    def get_schedule_data():
        with data_lock:
            if schedule_data is None:
                return jsonify({"status": "not_ready"})
            return jsonify({"status": "ready", "data": schedule_data})

    # --- REFINEMENT: Use consistent status envelope ---
    @app.route('/distribution_data')
    def get_distribution_data():
        with data_lock:
            return jsonify({"status": "ready", "data": timestep_counts.tolist()})

    # --- Data Listener Thread ---
    def data_listener():
        nonlocal schedule_data, settings_data
        for line in sys.stdin:
            # --- REFINEMENT: Debug print is now commented out ---
            # print(f"Plotter RAW_IN: {line!r}")
            line = line.strip()
            if not line: continue

            if line.startswith("SETTINGS::"):
                try:
                    with data_lock:
                        settings_data = json.loads(line[len("SETTINGS::"):])
                    print("Plotter: Received settings data.")
                except Exception as e:
                    print(f"Plotter: Error processing settings data: {e}")
                continue

            if line.startswith("SCHEDULE::"):
                try:
                    alphas_cumprod = np.fromstring(line[len("SCHEDULE::"):], sep=',')
                    signal_factor = np.sqrt(alphas_cumprod)
                    noise_factor = np.sqrt(1 - alphas_cumprod)
                    with data_lock:
                        schedule_data = {
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
                with data_lock:
                    unique, counts = np.unique(timesteps, return_counts=True)
                    valid_indices = unique < len(timestep_counts)
                    timestep_counts[unique[valid_indices]] += counts[valid_indices]
            except (ValueError, IndexError):
                continue

    listener_thread = threading.Thread(target=data_listener, daemon=True)
    listener_thread.start()

    print(f"\n\n ✨ Interactive Dashboard is running at: http://localhost:{port} ✨ \n\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Timestep Plotter Server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    run_plotter_server(args.port)