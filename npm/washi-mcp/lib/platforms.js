// Platforms washi ships prebuilt binaries for, keyed by `${process.platform}-${process.arch}`.
// Each one becomes an npm package `washi-mcp-<key>`; scripts/pack.js and the launcher both read this.
module.exports = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-arm64": "aarch64-unknown-linux-musl",
  "linux-x64": "x86_64-unknown-linux-musl",
  "win32-x64": "x86_64-pc-windows-msvc",
};
