use rmst_pliron_b::prepare::{self, Manifest};
use serde::Deserialize;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Accept {
        text: String,
    },
    Prepare {
        accepted: String,
        manifest: Manifest,
        capacity: usize,
        fuse: bool,
    },
    Load {
        accepted: String,
        prepared: String,
        manifest: Manifest,
    },
}
fn run() -> Result<serde_json::Value, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let request: Request = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    match request {
        Request::Accept { text } => prepare::accept(&text)
            .map(|text| serde_json::json!({"accepted": text, "pliron": "0.20.0"})),
        Request::Prepare {
            accepted,
            manifest,
            capacity,
            fuse,
        } => serde_json::to_value(prepare::prepare(&accepted, manifest, capacity, fuse)?)
            .map_err(|e| e.to_string()),
        Request::Load {
            accepted,
            prepared,
            manifest,
        } => serde_json::to_value(prepare::load(&accepted, &prepared, manifest)?)
            .map_err(|e| e.to_string()),
    }
}
fn main() {
    match run() {
        Ok(output) => println!("{}", serde_json::to_string(&output).unwrap()),
        Err(error) => {
            eprintln!("{}", serde_json::json!({"error": error}));
            std::process::exit(2);
        }
    }
}
