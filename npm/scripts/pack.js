#!/usr/bin/env node
// Builds the npm packages from compiled binaries:
//   dist/npm/washi-mcp             launcher, depends on all platform packages (optional)
//   dist/npm/washi-mcp-<platform>  the two native binaries for one platform (see packageName in platforms.js)
//
// usage: node npm/scripts/pack.js <bin-dir> [--allow-missing]
// <bin-dir>/<rust-target>/ holds washi and washi-mcp (with .exe on Windows), e.g. bins/aarch64-apple-darwin/washi
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const { targets, packageName } = require("../washi-mcp/lib/platforms.js");
const [binDir, ...flags] = process.argv.slice(2);
if (!binDir) {
  console.error("usage: node npm/scripts/pack.js <bin-dir> [--allow-missing]");
  process.exit(2);
}
const allowMissing = flags.includes("--allow-missing");

const cargo = fs.readFileSync(path.join(root, "Cargo.toml"), "utf8");
const version = cargo.match(/^version\s*=\s*"([^"]+)"/m)[1];
const out = path.join(root, "dist", "npm");
fs.rmSync(out, { recursive: true, force: true });

const main = JSON.parse(fs.readFileSync(path.join(root, "npm/washi-mcp/package.json"), "utf8"));
const write = (dir, file, data) => {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, file), typeof data === "string" ? data : JSON.stringify(data, null, 2) + "\n");
};

const built = [];
for (const [key, target] of Object.entries(targets)) {
  const [os, cpu] = key.split("-");
  const exe = os === "win32" ? ".exe" : "";
  const srcs = ["washi", "washi-mcp"].map((n) => path.join(binDir, target, n + exe));
  const missing = srcs.filter((s) => !fs.existsSync(s));
  if (missing.length) {
    if (!allowMissing) throw new Error(`missing binaries for ${key}: ${missing.join(", ")}`);
    console.warn(`skipping ${key} (no binaries in ${path.join(binDir, target)})`);
    continue;
  }
  const name = packageName(key);
  const dir = path.join(out, name);
  fs.mkdirSync(path.join(dir, "bin"), { recursive: true });
  for (const src of srcs) {
    const dest = path.join(dir, "bin", path.basename(src));
    fs.copyFileSync(src, dest);
    fs.chmodSync(dest, 0o755);
  }
  write(dir, "package.json", {
    name,
    version,
    description: `The ${key} binaries for ${main.name}. Install ${main.name} instead of this package.`,
    license: main.license,
    repository: main.repository,
    os: [os],
    cpu: [cpu],
    files: ["bin"],
    preferUnplugged: true,
  });
  fs.copyFileSync(path.join(root, "LICENSE"), path.join(dir, "LICENSE"));
  write(dir, "README.md", `# ${name}\n\nThe \`${target}\` binaries for [\`${main.name}\`](https://www.npmjs.com/package/${main.name}). Install that package instead.\n`);
  built.push(name);
}

const dir = path.join(out, main.name);
fs.cpSync(path.join(root, "npm/washi-mcp"), dir, { recursive: true });
fs.copyFileSync(path.join(root, "README.md"), path.join(dir, "README.md"));
fs.copyFileSync(path.join(root, "LICENSE"), path.join(dir, "LICENSE"));
write(dir, "package.json", {
  ...main,
  version,
  optionalDependencies: Object.fromEntries(Object.keys(targets).map((k) => [packageName(k), version])),
});

console.log(`version ${version}`);
for (const name of [...built, main.name]) console.log(`  dist/npm/${name}`);
