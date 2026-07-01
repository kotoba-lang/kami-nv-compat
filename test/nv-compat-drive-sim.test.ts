/**
 * nv-compat DRIVE Sim / wadachi-sim validation.
 *
 * Exercises the clean-room sensor-realistic AV simulator: the world + scene
 * construction, the camera / LiDAR / radar sensor models (grounded in the
 * kami-rt ray tracer), the open- and closed-loop step (closing the loop with
 * the Alpamayo VLA facade), and the USD → scenario bridge. Deterministic,
 * CPU-only.
 *
 *     pnpm exec vitest run test/nv-compat-drive-sim.test.ts
 *
 * ADR-2605261800 §D1/D6 (DriveSim → wadachi-sim); AV scope per wadachi /
 * kami-autodrive ADRs.
 */

import { describe, it, expect } from "vitest";
import {
  type Actor,
  type StaticObstacle,
  DriveSim,
  createScenario,
  createCamera,
  createLidar,
  createRadar,
  obstaclesFromStage,
} from "../src/drive-sim.js";
import {
  boxTris,
  buildSensorScene,
  groundTruth,
  sampleLidar,
  sampleRadar,
  sampleCamera,
  worldAabb,
} from "../src/wadachi-sim/index.js";
import { AlpamayoR1 } from "../src/alpamayo.js";
import { Stage } from "../src/omni-usd.js";

function obstacleAhead(x: number): StaticObstacle {
  return { id: "wall", kind: "vehicle", x, y: 0, yaw: 0, extent: [1, 1, 1] };
}

describe("wadachi-sim world", () => {
  it("tessellates an oriented box into 12 triangles", () => {
    expect(boxTris(0, 0, [1, 1, 1], 0)).toHaveLength(12);
  });

  it("worldAabb encloses a rotated box", () => {
    const { min, max } = worldAabb(5, 0, [2, 1, 0.75], Math.PI / 2);
    // 90° rotation swaps the x/y extents.
    expect(max[0] - min[0]).toBeCloseTo(2, 5); // was 2*hy = 2
    expect(max[1] - min[1]).toBeCloseTo(4, 5); // was 2*hx = 4
    expect(min[2]).toBe(0);
    expect(max[2]).toBeCloseTo(1.5, 5);
  });

  it("buildSensorScene includes ground + obstacles; groundTruth lists objects", () => {
    const scenario = createScenario({ obstacles: [obstacleAhead(20)] });
    const scene = buildSensorScene(scenario);
    expect(scene.soup.count).toBe(2 + 12); // ground (2) + 1 box (12)
    expect(groundTruth(scenario)).toHaveLength(1);
  });
});

describe("LiDAR sensor", () => {
  it("ranges a box directly ahead at the correct distance", () => {
    const scenario = createScenario({ ego: { speed: 0 }, obstacles: [obstacleAhead(20)] });
    const scene = buildSensorScene(scenario);
    // Single horizontal ray straight ahead.
    const scan = sampleLidar(scenario, scene, {
      azimuthFovDeg: 0,
      azimuthSteps: 1,
      elevationFovDeg: 0,
      elevationSteps: 1,
      maxRange: 80,
      mount: { forward: 1.5, left: 0, height: 1.5, yaw: 0 },
    });
    expect(scan.rays).toBe(1);
    expect(scan.returns).toHaveLength(1);
    // Box front face at x=19; sensor at x=1.5 → range 17.5.
    expect(scan.returns[0].range).toBeCloseTo(17.5, 4);
    expect(scan.returns[0].point[0]).toBeCloseTo(19, 4);
  });

  it("produces a point cloud over a fan of rays", () => {
    const scenario = createScenario({ ego: { speed: 0 }, obstacles: [obstacleAhead(15)] });
    const scan = sampleLidar(scenario, buildSensorScene(scenario), createLidar({ azimuthFovDeg: 60, azimuthSteps: 30, elevationSteps: 3, maxRange: 60 }));
    expect(scan.returns.length).toBeGreaterThan(0);
    for (const r of scan.returns) expect(r.range).toBeLessThanOrEqual(60);
  });
});

describe("radar sensor", () => {
  it("reports range, azimuth and approaching range-rate for a closing actor", () => {
    const actor: Actor = { id: "car", kind: "vehicle", x: 30, y: 0, yaw: 0, vx: -5, vy: 0, extent: [2, 1, 0.75] };
    const scenario = createScenario({ ego: { speed: 0 }, actors: [actor] });
    const dets = sampleRadar(scenario, createRadar({ azimuthFovDeg: 120, maxRange: 150 }));
    expect(dets).toHaveLength(1);
    expect(dets[0].id).toBe("car");
    expect(dets[0].range).toBeCloseTo(28.5, 4); // 30 − sensor x(1.5)
    expect(Math.abs(dets[0].azimuth)).toBeLessThan(0.01);
    expect(dets[0].rangeRate).toBeLessThan(0); // closing
  });

  it("drops actors outside the azimuth FOV", () => {
    const side: Actor = { id: "side", kind: "vehicle", x: 0, y: 30, yaw: 0, vx: 0, vy: 0, extent: [2, 1, 0.75] };
    const dets = sampleRadar(createScenario({ ego: { speed: 0 }, actors: [side] }), createRadar({ azimuthFovDeg: 60 }));
    expect(dets).toHaveLength(0); // 90° off-axis, outside ±30°
  });
});

describe("camera sensor", () => {
  it("renders a frame and projects a ground-truth box for an actor ahead", () => {
    const actor: Actor = { id: "lead", kind: "vehicle", x: 15, y: 0, yaw: 0, vx: 0, vy: 0, extent: [2, 1, 0.75] };
    const scenario = createScenario({ ego: { speed: 0 }, actors: [actor] });
    const cam = sampleCamera(scenario, groundTruth(scenario), buildSensorScene(scenario), createCamera({ width: 64, height: 36, vfovDeg: 50 }));
    expect(cam.rgb.length).toBe(64 * 36 * 4);
    const box = cam.boxes.find((b) => b.id === "lead");
    expect(box).toBeDefined();
    const [x, y, w, h] = box!.bbox2d;
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x + w).toBeLessThanOrEqual(64);
    expect(y + h).toBeLessThanOrEqual(36);
    expect(w).toBeGreaterThan(0);
  });
});

describe("DriveSim step loop + closed-loop driving", () => {
  it("advances the ego forward in open loop", () => {
    const sim = new DriveSim({ scenario: createScenario({ ego: { speed: 8 } }), rig: {}, hz: 10 });
    const x0 = sim.world.ego.x;
    for (let i = 0; i < 10; i++) sim.step({ action: { accel: 0, curvature: 0 } });
    expect(sim.world.ego.x).toBeGreaterThan(x0);
    expect(sim.world.ego.x).toBeCloseTo(8 * 1.0, 1); // ~8 m in 1 s
  });

  it("closes the loop with Alpamayo: ego decelerates for a pedestrian ahead", () => {
    const ped: Actor = { id: "ped", kind: "pedestrian", x: 14, y: 0, yaw: 0, vx: 0, vy: 0, extent: [0.3, 0.3, 0.9] };
    const sim = new DriveSim({
      scenario: createScenario({ ego: { speed: 8 }, actors: [ped] }),
      rig: { radar: createRadar() },
      hz: 10,
      command: "keep_lane",
    });
    const model = AlpamayoR1.fromPretrained();
    const startSpeed = sim.world.ego.speed;
    const frames = sim.run(model, 20);
    expect(frames).toHaveLength(20);
    expect(frames[0].radar).toBeDefined();
    expect(sim.world.ego.speed).toBeLessThan(startSpeed); // braked for the pedestrian
  });

  it("reset + rerun is deterministic", () => {
    const make = () =>
      new DriveSim({ scenario: createScenario({ ego: { speed: 7 }, actors: [{ id: "a", kind: "vehicle", x: 40, y: 0, yaw: 0, vx: 0, vy: 0, extent: [2, 1, 0.75] }] }), rig: {}, hz: 10 });
    const model = AlpamayoR1.fromPretrained();
    const a = make();
    a.run(model, 15);
    const xa = a.world.ego.x;
    const b = make();
    b.run(model, 15);
    expect(b.world.ego.x).toBe(xa);
    // In-place reset reproduces the same trajectory.
    a.reset();
    expect(a.tick).toBe(0);
    a.run(model, 15);
    expect(a.world.ego.x).toBe(xa);
  });
});

describe("USD → DriveSim scenario bridge", () => {
  it("builds box obstacles from a USD stage's meshes", () => {
    const usda = `#usda 1.0
def Xform "World" {
    def Mesh "wall" {
        point3f[] points = [(9,-1,0),(11,-1,0),(11,1,0),(9,1,0),(9,-1,2),(11,-1,2),(11,1,2),(9,1,2)]
        int[] faceVertexCounts = [4,4]
        int[] faceVertexIndices = [0,1,2,3,4,5,6,7]
    }
}`;
    const obstacles = obstaclesFromStage(Stage.Open(usda));
    expect(obstacles.length).toBeGreaterThan(0);
    // The mesh spans x∈[9,11] → center x ≈ 10.
    expect(obstacles[0].x).toBeCloseTo(10, 5);
  });
});
