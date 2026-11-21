// Finds the native washi binary for this machine and runs it with the same arguments and stdio.
const { spawn } = require("node:child_process");
const { constants } = require("node:os");
const { targets, packageName } = require("./platforms.js");

const key = `${process.platform}-${process.arch}`;

/** Path of the native binary (`washi` or `washi-mcp`) from the platform package. */
function binaryPath(name) {
  if (!targets[key]) {
    throw new Error(
      `no prebuilt binary for ${key} (available: ${Object.keys(targets).join(", ")}). ` +
        "You can build it from source with `cargo install --git https://github.com/Idered/washi`.",
    );
  }
  const pkg = packageName(key);
  const exe = process.platform === "win32" ? `${name}.exe` : name;
  try {
    return require.resolve(`${pkg}/bin/${exe}`);
  } catch {
    throw new Error(
      `the package ${pkg} is missing. npm installs it automatically as an optional dependency; ` +
        "reinstall without --omit=optional / --no-optional.",
    );
  }
}

function run(name) {
  let bin;
  try {
    bin = binaryPath(name);
  } catch (e) {
    console.error(`${name}: ${e.message}`);
    process.exit(1);
  }
  const child = spawn(bin, process.argv.slice(2), { stdio: "inherit" });
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, () => child.kill(signal));
  }
  child.on("error", (e) => {
    console.error(`${name}: ${e.message}`);
    process.exit(1);
  });
  child.on("exit", (code, signal) => {
    process.exit(signal ? 128 + (constants.signals[signal] ?? 0) : (code ?? 1));
  });
}

module.exports = { binaryPath, run, key };
