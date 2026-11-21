// Platforms washi ships prebuilt binaries for, keyed by `${process.platform}-${process.arch}`.
// Each one becomes its own npm package; scripts/pack.js and the launcher both read this.
const targets = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-arm64": "aarch64-unknown-linux-musl",
  "linux-x64": "x86_64-unknown-linux-musl",
  "win32-x64": "x86_64-pc-windows-msvc",
};

/** npm package holding the binaries for a platform key, e.g. washi-mcp-darwin-arm64. */
function packageName(key) {
  // the registry refuses to stage washi-mcp-win32-x64, so the Windows package says "windows"
  return `washi-mcp-${key.replace(/^win32-/, "windows-")}`;
}

module.exports = { targets, packageName };
