//! Talks to the real `washi-mcp` binary over stdio, the way an MCP client does.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
    dir: PathBuf,
}

impl Server {
    /// Starts the server with its own output folder and completes the handshake.
    fn start(test: &str, protocol: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("washi-test-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_washi-mcp"))
            .env("WASHI_OUT", &dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("start washi-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut s = Server { child, stdin, stdout, next_id: 1, dir };
        let init = s.request("initialize", json!({"protocolVersion": protocol, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}));
        assert_eq!(init["result"]["protocolVersion"], protocol);
        s.notify("notifications/initialized");
        s
    }

    fn send(&mut self, msg: Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn notify(&mut self, method: &str) {
        self.send(json!({"jsonrpc": "2.0", "method": method}));
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad JSON from server ({e}): {line}"));
        assert_eq!(reply["id"], id, "reply to the wrong request: {reply}");
        reply
    }

    /// Calls a tool and returns its result (not a JSON-RPC error).
    fn call(&mut self, tool: &str, args: Value) -> Value {
        let reply = self.request("tools/call", json!({"name": tool, "arguments": args}));
        reply.get("result").cloned().unwrap_or_else(|| panic!("{tool} returned a protocol error: {reply}"))
    }

    fn error_text(&mut self, tool: &str, args: Value) -> String {
        let r = self.call(tool, args.clone());
        assert_eq!(r["isError"], true, "{tool} {args} should fail: {r}");
        r["content"][0]["text"].as_str().unwrap().to_string()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn is_png(path: &Path) -> bool {
    std::fs::read(path).is_ok_and(|b| b.starts_with(b"\x89PNG\r\n\x1a\n"))
}

fn content_of<'a>(result: &'a Value, kind: &str) -> Vec<&'a Value> {
    result["content"].as_array().unwrap().iter().filter(|c| c["type"] == kind).collect()
}

#[test]
fn tool_definitions_follow_best_practices() {
    let mut s = Server::start("schema", "2025-06-18");
    let tools = s.request("tools/list", json!({}))["result"]["tools"].clone();
    let tools = tools.as_array().unwrap();
    assert_eq!(tools.len(), 5);
    for t in tools {
        let name = t["name"].as_str().unwrap();
        assert!(t["title"].is_string(), "{name}: title");
        let desc = t["description"].as_str().unwrap();
        assert!(desc.len() < 300, "{name}: description should be short, details belong on parameters");
        let schema = &t["inputSchema"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false, "{name}: additionalProperties");
        for (prop, def) in schema["properties"].as_object().unwrap() {
            assert!(def["description"].is_string(), "{name}.{prop}: missing description");
        }
        let a = &t["annotations"];
        assert_eq!(a["openWorldHint"], false, "{name}: openWorldHint");
        let read_only = name.starts_with("list_");
        assert_eq!(a["readOnlyHint"], read_only, "{name}: readOnlyHint");
        assert_eq!(t.get("outputSchema").is_some(), read_only, "{name}: outputSchema");
    }
    let gen_props = &tools[0]["inputSchema"]["properties"];
    assert_eq!(gen_props["width"]["minimum"], 16);
    assert_eq!(gen_props["width"]["maximum"], 8192);
    assert_eq!(gen_props["shade_strength"]["maximum"], 1.0);
    assert_eq!(gen_props["grain_size"]["exclusiveMinimum"], 0.0);
}

#[test]
fn list_tools_return_structured_content() {
    let mut s = Server::start("lists", "2025-06-18");
    let r = s.call("list_palettes", json!({}));
    let palettes = r["structuredContent"].as_object().unwrap();
    assert!(palettes.len() >= 10);
    for (name, p) in palettes {
        assert!(p["description"].is_string(), "{name}");
        assert!(p["colors"].as_array().is_some_and(|c| c.len() >= 2), "{name}");
    }
    // the same JSON as text, for clients that don't read structuredContent
    let text: Value = serde_json::from_str(r["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(&text, &r["structuredContent"]);

    let r = s.call("list_types", json!({}));
    assert_eq!(r["structuredContent"]["aurora"]["tonemap"], "aces");
}

#[test]
fn generate_saves_png_and_links_it() {
    let mut s = Server::start("generate", "2025-06-18");
    let args = json!({"type": "aurora", "palette": "borealis", "width": 160, "height": 90});
    let r = s.call("generate_gradient", args.clone());
    assert_eq!(r["isError"], false, "{r}");
    assert!(r["content"][0]["text"].as_str().unwrap().starts_with("Saved "));
    let link = content_of(&r, "resource_link");
    assert_eq!(link.len(), 1);
    assert_eq!(link[0]["mimeType"], "image/png");
    let uri = link[0]["uri"].as_str().unwrap();
    assert!(uri.starts_with("file:///"), "{uri}");
    assert_eq!(content_of(&r, "image")[0]["mimeType"], "image/jpeg");

    let pngs = |dir: &Path| std::fs::read_dir(dir).unwrap().filter(|e| is_png(&e.as_ref().unwrap().path())).count();
    assert_eq!(pngs(&s.dir), 1);
    // the same call again, within the same second, must not replace the first file
    s.call("generate_gradient", args);
    assert_eq!(pngs(&s.dir), 2);

    let r = s.call("generate_gradient", json!({"width": 64, "height": 36, "preview": false, "output_path": "sub/a.png"}));
    assert!(content_of(&r, "image").is_empty());
    assert!(is_png(&s.dir.join("sub/a.png")), "relative paths go inside the output folder");
}

#[test]
fn older_clients_get_no_resource_links() {
    let mut s = Server::start("old-client", "2025-03-26");
    let r = s.call("generate_gradient", json!({"width": 64, "height": 36}));
    assert!(content_of(&r, "resource_link").is_empty());
    assert_eq!(content_of(&r, "image").len(), 1);
}

#[test]
fn bad_arguments_are_reported_as_tool_errors() {
    let mut s = Server::start("errors", "2025-06-18");
    assert!(s.error_text("generate_gradient", json!({"colour": "#fff"})).contains("unknown argument \"colour\""));
    assert!(s.error_text("list_types", json!({"x": 1})).contains("takes no arguments"));
    assert!(s.error_text("generate_gradient", json!({"width": 5})).contains("16..8192"));
    assert!(s.error_text("generate_gradient", json!({"noise": 150})).contains("noise must be between 0 and 100"));
    assert!(s.error_text("generate_gradient", json!({"grain": -1})).contains("grain must be at least 0"));
    assert!(s.error_text("generate_gradient", json!({"type": "nope"})).contains("invalid argument \"type\""));
    assert!(s.error_text("preview_seeds", json!({"seeds": [-1]})).contains("seeds"));
}

#[test]
fn output_path_cannot_clobber_other_files() {
    let mut s = Server::start("paths", "2025-06-18");
    let small = |path: &str| json!({"width": 64, "height": 36, "preview": false, "output_path": path});

    assert!(s.error_text("generate_gradient", small("notes.txt")).contains("must end in .png"));

    let fake = s.dir.join("fake.png");
    std::fs::write(&fake, "not an image").unwrap();
    let err = s.error_text("generate_gradient", small(fake.to_str().unwrap()));
    assert!(err.contains("isn't a PNG"), "{err}");
    assert_eq!(std::fs::read_to_string(&fake).unwrap(), "not an image");

    std::fs::create_dir_all(s.dir.join("folder.png")).unwrap();
    assert!(s.error_text("generate_gradient", small("folder.png")).contains("is a folder"));

    // an existing PNG may be replaced (iterating on one output file)
    let real = s.dir.join("real.png");
    assert_eq!(s.call("generate_gradient", small(real.to_str().unwrap()))["isError"], false);
    assert_eq!(s.call("generate_gradient", small(real.to_str().unwrap()))["isError"], false);
}

#[test]
fn protocol_errors_and_notifications() {
    let mut s = Server::start("protocol", "2025-06-18");
    let r = s.request("tools/call", json!({"name": "nope", "arguments": {}}));
    assert_eq!(r["error"]["code"], -32602);
    let r = s.request("bogus/method", json!({}));
    assert_eq!(r["error"]["code"], -32601);
    // notifications get no reply, so the next line read must be the ping's answer
    s.notify("notifications/cancelled");
    assert_eq!(s.request("ping", json!({}))["result"], json!({}));
}
