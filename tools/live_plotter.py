import sys
import threading
import json
import argparse
import os

import numpy as np
from flask import Flask, jsonify, send_from_directory


def run_plotter_server(port):
    # These are our in-memory "databases"
    timestep_counts = np.zeros(1000, dtype=np.int64)
    schedule_data = None
    data_lock = threading.Lock()

    # --- Flask App Setup ---
    # We tell it where to find your dashboard.html file
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    app = Flask(__name__, static_folder=web_dir)

    # --- API and File Serving Routes ---
    @app.route('/')
    def index():
        # Serves your beautiful HTML file!
        return send_from_directory(app.static_folder, 'dashboard.html')

    @app.route('/schedule_data')
    def get_schedule_data():
        with data_lock:
            if schedule_data is None:
                return jsonify({"error": "Schedule data not yet received"}), 404
            return jsonify(schedule_data)

    @app.route('/distribution_data')
    def get_distribution_data():
        with data_lock:
            # Send the raw counts array directly to the browser!
            return jsonify(timestep_counts.tolist())

    # --- Data Listener Thread (This part doesn't change) ---
    def data_listener():
        nonlocal schedule_data
        for line in sys.stdin:
            line = line.strip()
            if not line: continue

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
                    timestep_counts[unique] += counts
            except (ValueError, IndexError):
                continue

    listener_thread = threading.Thread(target=data_listener, daemon=True)
    listener_thread.start()

    print(f"\n\n ✨ Interactive Dashboard is running at: http://localhost:{port} ✨ \n\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


# --- Main execution block (This part doesn't change) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Timestep Plotter Server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # We remove the extra args because this script no longer saves files
    run_plotter_server(args.port)