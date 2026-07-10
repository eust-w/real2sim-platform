import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

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
  const fileBytes = new Uint8Array(await fs.readFile(input));
  const reader = new spark.PlyReader({ fileBytes });
  await reader.parseHeader();
  await reader.parseData(() => {});
  const numSplats = Number(reader.numSplats || 0);
  if (!numSplats) throw new Error("Spark PlyReader returned zero splats");

  const writer = new spark.SpzWriter({ numSplats, shDegree: 0 });
  let written = 0;
  reader.parseSplats((index, x, y, z, scaleX, scaleY, scaleZ, quatX, quatY, quatZ, quatW, opacity, red, green, blue) => {
    writer.setCenter(index, x, y, z);
    writer.setScale(index, Math.max(scaleX, 1e-6), Math.max(scaleY, 1e-6), Math.max(scaleZ, 1e-6));
    writer.setQuat(index, quatX, quatY, quatZ, quatW);
    writer.setAlpha(index, Math.max(0, Math.min(1, Number(opacity ?? 1))));
    writer.setRgb(index, Math.max(0, Math.min(1, red)), Math.max(0, Math.min(1, green)), Math.max(0, Math.min(1, blue)));
    written += 1;
  });
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
