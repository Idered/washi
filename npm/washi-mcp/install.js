// Replaces the Node launchers in bin/ with the native binaries, so running washi-mcp doesn't keep
// an extra Node process (~40 MB) alive next to a 2 MB server. If anything goes wrong, or install
// scripts are disabled, the launchers stay and still work.
const fs = require("node:fs");
const path = require("node:path");

if (process.platform === "win32") process.exit(0); // npm runs bins through .cmd shims there

let binaryPath;
try {
  ({ binaryPath } = require("./lib/run.js"));
} catch {
  process.exit(0);
}

for (const name of ["washi-mcp", "washi"]) {
  try {
    const src = binaryPath(name);
    const dest = path.join(__dirname, "bin", name);
    const tmp = `${dest}.tmp-${process.pid}`;
    try {
      fs.linkSync(src, tmp);
    } catch {
      fs.copyFileSync(src, tmp);
    }
    fs.chmodSync(tmp, 0o755);
    fs.renameSync(tmp, dest);
  } catch {
    // keep the launcher
  }
}
