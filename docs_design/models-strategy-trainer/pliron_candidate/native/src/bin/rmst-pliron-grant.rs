#[path = "../grant/mod.rs"]
mod grant;

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
        manifest: grant::Manifest,
    },
    Load {
        accepted: String,
        prepared: String,
        manifest: grant::Manifest,
    },
}
fn run() -> Result<serde_json::Value, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    match serde_json::from_str::<Request>(&input).map_err(|e| e.to_string())? {
        Request::Accept { text } => grant::accept(&text)
            .map(|accepted| serde_json::json!({"accepted":accepted,"pliron":"0.20.0"})),
        Request::Prepare { accepted, manifest } => grant::prepare(&accepted, None, manifest),
        Request::Load {
            accepted,
            prepared,
            manifest,
        } => grant::prepare(&accepted, Some(&prepared), manifest),
    }
}
fn main() {
    match run() {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("{}", serde_json::json!({"error":error}));
            std::process::exit(2);
        }
    }
}
