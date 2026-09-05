<script>
import { onMount } from "svelte";

let variants = [];
let selectedVariant = "";
let selectedDay = "";
let loading = true;
let busy = false;
let status = "";
let error = false;
let fileInput;

$: current = variants.find((v) => v.variant === selectedVariant);
$: days = current?.days ?? [];

async function loadOptions(preferredVariant = null) {
  loading = true;
  try {
    const res = await fetch("/api/strategy/options", { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Could not load strategy options");
    variants = data.variants ?? [];
    const wanted = preferredVariant && variants.some(v => v.variant === preferredVariant)
      ? preferredVariant
      : selectedVariant && variants.some(v => v.variant === selectedVariant)
        ? selectedVariant
        : variants[0]?.variant ?? "";
    selectedVariant = wanted;
    const v = variants.find(x => x.variant === wanted);
    if (!v?.days?.some(d => String(d.day) === String(selectedDay))) {
      selectedDay = v?.days?.[0]?.day != null ? String(v.days[0].day) : "";
    }
  } catch (e) {
    error = true;
    status = e.message || "Could not load strategy options";
  } finally { loading = false; }
}

async function uploadFile() {
  const file = fileInput?.files?.[0];
  if (!file) return;
  busy = true; error = false; status = `Uploading ${file.name}...`;
  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/strategy/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    await loadOptions(data.variant);
    status = `Uploaded ${data.filename}. Select a day and click Apply.`;
  } catch (e) { error = true; status = e.message || "Upload failed"; }
  finally { busy = false; if (fileInput) fileInput.value = ""; }
}

async function applyStrategy() {
  if (!selectedVariant || selectedDay === "") return;
  busy = true; error = false; status = "Applying offline model...";
  try {
    const res = await fetch("/api/strategy/push", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: selectedVariant, day: Number(selectedDay) })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Push failed");
    status = `Applied ${selectedVariant}, Day ${selectedDay} — ${data.points} points.`;
  } catch (e) { error = true; status = e.message || "Push failed"; }
  finally { busy = false; }
}

onMount(() => loadOptions());
</script>

<div class="metric-card svelte-14ry3bj p-4 mb-4">
  <div class="flex items-center justify-between mb-3">
    <h3 class="text-lg font-semibold">Offline Model Strategy</h3>
    <span class="text-sm text-gray-400">Upload → select day → apply</span>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
    <div>
      <label class="text-sm text-gray-400 block mb-1">Offline model output (.json)</label>
      <input bind:this={fileInput} type="file" accept=".json,application/json" onchange={uploadFile}
        class="w-full text-sm text-gray-300 bg-gray-800 rounded border border-gray-600 p-1.5" />
    </div>
    <div>
      <label class="text-sm text-gray-400 block mb-1">Strategy</label>
      <select bind:value={selectedVariant} class="w-full bg-gray-800 text-white rounded px-2 py-1.5 border border-gray-600">
        {#if variants.length === 0}<option value="">No strategies</option>{/if}
        {#each variants as v}<option value={v.variant}>{v.variant}</option>{/each}
      </select>
    </div>
    <div>
      <label class="text-sm text-gray-400 block mb-1">Day</label>
      <select bind:value={selectedDay} class="w-full bg-gray-800 text-white rounded px-2 py-1.5 border border-gray-600">
        {#each days as d}<option value={String(d.day)}>Day {d.day} — {d.route ?? "?"}</option>{/each}
      </select>
    </div>
    <button onclick={applyStrategy} disabled={busy || !selectedVariant || selectedDay === ""}
      class="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded px-4 py-1.5 font-medium">
      {busy ? "Working..." : "Apply to Dashboard"}
    </button>
  </div>
  {#if loading}<p class="mt-2 text-sm text-gray-400">Loading strategies...</p>{/if}
  {#if status}<p class={`mt-2 text-sm ${error ? "text-red-400" : "text-green-400"}`}>{status}</p>{/if}
</div>
