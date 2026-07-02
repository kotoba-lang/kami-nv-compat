# kami-nv-compat

Drop-in NVIDIA Omniverse / Isaac Sim / Isaac Lab / OptiX / RTX Renderer /
Replicator / DriveSim / Omniverse Cloud / Nucleus / Alpamayo API-compat facade
for the KAMI engine family, targeting WebGPU + WASM. See
[`src/README.md`](./src/README.md) for the full module map, trademark notice,
and R1 sub-phase delivery status.

## Provenance

Relocated 2026-07-01 from `etzhayyim/root:20-actors/etzhayyim-sdk/src/nv-compat/`
to `kotoba-lang/kami-nv-compat` per the org-taxonomy library-placement rule
(any library/substrate code belongs in `kotoba-lang`, ADR-2606302300). Design
authority remains ADR-2605261800 (etzhayyim/root). `@etzhayyim/sdk` now consumes
this package as an external dependency instead of vendoring it under `src/`.

Canonical KAMI engines this facade mirrors (`amenominaka`, `e7m-sim`,
`e7m-shugyo`, `hikari-rt`/`kami-rt`, `kami-rtx`, `utsushimi`, `wadachi-sim`,
`murakumo-render`, `kotoba-datomic-nucleus`, `michibiki`/`kami-drive`) are
implemented in-tree here (portable `.cljc` + WGSL, per the kotoba-lang layer
test — see ADR-2607020130); the longer-term plan is to back them with the
Rust crates under the sibling `kotoba-lang/kami-engine` repo (`kami-genesis`,
`kami-articulated`, `kami-rt`, `kami-usd`, …) once those land past
path-reservation — see ADR-2605261800 §D6/§D7.

## Development

```bash
clojure -M:lint   # clj-kondo
clojure -M:test   # cognitect test-runner
```

## License

Apache 2.0 + Charter Compliance Rider v3.6 (`/CHARTER-RIDER.md`).
