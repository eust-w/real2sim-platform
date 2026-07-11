import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

const SH_C0 = 0.28209479177387814;

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) continue;
    const key = item.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) args[key] = true;
    else {
      args[key] = next;
      index += 1;
    }
  }
  return args;
}

async function ensureSparkImport(vendorRoot) {
  const tmpRoot = path.join("/tmp", `real2sim-spark-node-${process.pid}`);
  const threeRoot = path.join(tmpRoot, "node_modules", "three");
  await fs.mkdir(path.join(threeRoot, "addons", "postprocessing"), { recursive: true });
  await fs.writeFile(
    path.join(threeRoot, "package.json"),
    JSON.stringify({ name: "three", type: "module", main: "./three.module.js", exports: { ".": "./three.module.js", "./addons/*": "./addons/*" } }),
  );
  await fs.copyFile(path.join(vendorRoot, "three", "three.module.js"), path.join(threeRoot, "three.module.js"));
  await fs.copyFile(path.join(vendorRoot, "three", "three.core.js"), path.join(threeRoot, "three.core.js"));
  await fs.copyFile(path.join(vendorRoot, "three", "addons", "postprocessing", "Pass.js"), path.join(threeRoot, "addons", "postprocessing", "Pass.js"));
  await fs.copyFile(path.join(vendorRoot, "spark.module.min.js"), path.join(tmpRoot, "spark.module.min.js"));
  return import(pathToFileURL(path.join(tmpRoot, "spark.module.min.js")).href);
}

async function readPlyHeader(input) {
  const handle = await fs.open(input, "r");
  try {
    const buffer = Buffer.alloc(4 * 1024 * 1024);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    const text = buffer.subarray(0, bytesRead).toString("utf8");
    const end = text.indexOf("end_header");
    if (end < 0) throw new Error("PLY header exceeds 4 MiB or is missing end_header");
    const lines = text.slice(0, end + "end_header".length).split(/\r?\n/);
    let format = "";
    let activeElement = "";
    let numSplats = 0;
    const properties = [];
    for (const rawLine of lines) {
      const parts = rawLine.trim().split(/\s+/);
      if (parts[0] === "format") format = parts[1] || "";
      if (parts[0] === "element") {
        activeElement = parts[1] || "";
        if (activeElement === "vertex") numSplats = Number(parts[2] || 0);
      }
      if (parts[0] === "property" && activeElement === "vertex") {
        if (parts[1] === "list") throw new Error("list properties are not supported for Gaussian vertices");
        properties.push(parts.at(-1));
      }
    }
    if (!format || !numSplats || !properties.length) throw new Error("invalid Gaussian PLY header");
    return { format, numSplats, properties };
  } finally {
    await handle.close();
  }
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

function sigmoid(value) {
  const number = Number(value);
  if (number >= 0) return 1 / (1 + Math.exp(-number));
  const exponential = Math.exp(number);
  return exponential / (1 + exponential);
}

function normalizeQuaternion(w, x, y, z) {
  const length = Math.hypot(w, x, y, z) || 1;
  return [x / length, y / length, z / length, w / length];
}

async function convertAsciiPly(input, header, spark) {
  const propertyIndex = new Map(header.properties.map((name, index) => [name, index]));
  const required = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
  const missing = required.filter((name) => !propertyIndex.has(name));
  if (missing.length) throw new Error(`ASCII Gaussian PLY is missing properties: ${missing.join(", ")}`);
  const hasShColor = ["f_dc_0", "f_dc_1", "f_dc_2"].every((name) => propertyIndex.has(name));
  const hasRgbColor = ["red", "green", "blue"].every((name) => propertyIndex.has(name));
  if (!hasShColor && !hasRgbColor) throw new Error("ASCII Gaussian PLY has neither f_dc_* nor RGB color properties");

  const writer = new spark.SpzWriter({ numSplats: header.numSplats, shDegree: 0 });
  const reader = createInterface({ input: createReadStream(input, { encoding: "utf8" }), crlfDelay: Infinity });
  let headerComplete = false;
  let written = 0;
  for await (const rawLine of reader) {
    const line = rawLine.trim();
    if (!headerComplete) {
      headerComplete = line === "end_header";
      continue;
    }
    if (!line || written >= header.numSplats) continue;
    const fields = line.split(/\s+/);
    if (fields.length < header.properties.length) throw new Error(`vertex ${written} has ${fields.length} values; expected ${header.properties.length}`);
    const value = (name) => Number(fields[propertyIndex.get(name)]);
    const numbers = required.map(value);
    if (numbers.some((number) => !Number.isFinite(number))) throw new Error(`vertex ${written} contains a non-finite required value`);
    const [quatX, quatY, quatZ, quatW] = normalizeQuaternion(value("rot_0"), value("rot_1"), value("rot_2"), value("rot_3"));
    writer.setCenter(written, value("x"), value("y"), value("z"));
    writer.setScale(
      written,
      Math.max(Math.exp(Math.max(-20, Math.min(20, value("scale_0")))), 1e-6),
      Math.max(Math.exp(Math.max(-20, Math.min(20, value("scale_1")))), 1e-6),
      Math.max(Math.exp(Math.max(-20, Math.min(20, value("scale_2")))), 1e-6),
    );
    writer.setQuat(written, quatX, quatY, quatZ, quatW);
    writer.setAlpha(written, propertyIndex.has("opacity") ? sigmoid(value("opacity")) : 1);
    if (hasShColor) {
      writer.setRgb(
        written,
        clamp01(0.5 + SH_C0 * value("f_dc_0")),
        clamp01(0.5 + SH_C0 * value("f_dc_1")),
        clamp01(0.5 + SH_C0 * value("f_dc_2")),
      );
    } else {
      const divisor = Math.max(value("red"), value("green"), value("blue")) > 1 ? 255 : 1;
      writer.setRgb(written, clamp01(value("red") / divisor), clamp01(value("green") / divisor), clamp01(value("blue") / divisor));
    }
    written += 1;
  }
  if (written !== header.numSplats) throw new Error(`ASCII Gaussian PLY declared ${header.numSplats} vertices but yielded ${written}`);
  return { writer, numSplats: header.numSplats, written };
}

async function main() {
  const args = parseArgs(process.argv);
  const input = args.input;
  const output = args.output;
  const vendorRoot = args["vendor-root"];
  const reportPath = args.report;
  if (!input || !output || !vendorRoot || !reportPath) {
    throw new Error("usage: node convert_3dgs_ply_to_spz.mjs --input scene.ply --output scene.spz --vendor-root static/vendor --report report.json");
  }

  const started = Date.now();
  const spark = await ensureSparkImport(vendorRoot);
  const header = await readPlyHeader(input);
  let writer;
  let numSplats;
  let written;
  if (header.format === "ascii") {
    ({ writer, numSplats, written } = await convertAsciiPly(input, header, spark));
  } else {
    if (header.format !== "binary_little_endian") throw new Error(`unsupported PLY format: ${header.format}`);
    const fileBytes = new Uint8Array(await fs.readFile(input));
    const reader = new spark.PlyReader({ fileBytes });
    await reader.parseHeader();
    await reader.parseData(() => {});
    numSplats = Number(reader.numSplats || 0);
    if (!numSplats) throw new Error("Spark PlyReader returned zero splats");
    writer = new spark.SpzWriter({ numSplats, shDegree: 0 });
    written = 0;
    reader.parseSplats((index, x, y, z, scaleX, scaleY, scaleZ, quatX, quatY, quatZ, quatW, opacity, red, green, blue) => {
      writer.setCenter(index, x, y, z);
      writer.setScale(index, Math.max(scaleX, 1e-6), Math.max(scaleY, 1e-6), Math.max(scaleZ, 1e-6));
      writer.setQuat(index, quatX, quatY, quatZ, quatW);
      writer.setAlpha(index, clamp01(opacity ?? 1));
      writer.setRgb(index, clamp01(red), clamp01(green), clamp01(blue));
      written += 1;
    });
  }
  const encoded = await writer.finalize();
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, encoded);
  const inputStat = await fs.stat(input);
  const outputStat = await fs.stat(output);
  const report = {
    schema: "real2sim.spz_export_report.v1",
    input,
    output,
    splats: numSplats,
    written_splats: written,
    input_format: header.format,
    input_size_bytes: inputStat.size,
    output_size_bytes: outputStat.size,
    compression_ratio: Number((outputStat.size / Math.max(inputStat.size, 1)).toFixed(4)),
    elapsed_ms: Date.now() - started,
    clipped_count: writer.clippedCount || 0,
  };
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report));
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
