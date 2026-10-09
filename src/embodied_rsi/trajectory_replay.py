"""Persist actual poses; render archived replay independently of the collecting arm."""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .storage import RUN_ID, write_json


class PoseRecorder:
    def __init__(self):
        self.positions, self.times, self.steps, self.closed = [], [], [], []
        self.step = -1

    def capture(self, world):
        self.positions.append(world.data.qpos.copy())
        self.times.append(float(world.data.time))
        self.steps.append(self.step)
        self.closed.append(world.closed)

    def save(self, directory, result, source="captured_poses"):
        directory.mkdir(parents=True, exist_ok=True)
        key = result["episode_id"]
        path = directory / f"{key}.npz"
        temporary = directory / f"{key}.npz.tmp"
        with temporary.open("wb") as file:
            np.savez_compressed(file, qpos=np.asarray(self.positions), times=self.times,
                                steps=self.steps, closed=self.closed)
        temporary.replace(path)
        meta = {"episode_id": key, "source": source, "frame_count": len(self.positions),
                "times": self.times, "steps": self.steps, "scene_hash": result["scene_hash"],
                "success": result["success"], "failure": result.get("failure"),
                "actions": [{"step": r["step"], "choice": r["choice"], "executed": r["executed"],
                             "rejection": r.get("rejection")} for r in result["records"]]}
        write_json(directory / f"{key}.json", meta)
        return {"source": source, "frame_count": len(self.positions)}


class TrajectoryReplay:
    def __init__(self, store):
        self.store = store
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archived-replay")
        self.key, self.world, self.poses, self.renderer = None, None, None, None

    def _episode(self, run, key):
        if not RUN_ID.fullmatch(key):
            raise ValueError("Invalid replay id")
        path = self.store.run_dir(run) / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError("采集批次不存在")
        with path.open() as file:
            for line in file:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                if row.get("episode_id") == key:
                    return row
        raise FileNotFoundError("轨迹不存在")

    def catalogue(self, run, outcome="all", offset=0, limit=5):
        path = self.store.run_dir(run) / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError("采集批次不存在")
        rows, count = [], 0
        with path.open() as file:
            for line in file:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                if outcome != "all" and bool(row["success"]) != (outcome == "success"):
                    continue
                if count >= offset and len(rows) < limit:
                    rows.append({k: v for k, v in row.items() if k != "records"})
                count += 1
        return {"episodes": rows, "total": count, "offset": offset}

    def _load(self, run, key):
        if self.key == (run, key):
            return
        import mujoco
        from embodied_jev.physics import RobotWorld

        result = self._episode(run, key)
        if result.get("success_definition") == "strict_grasp_v1":
            from .grasp import GraspWorld
            factory = GraspWorld
        else:
            factory = RobotWorld
        world = factory(result["task"], result["seed"], result.get("scene_config"))
        if world.scene_hash != result["scene_hash"]:
            raise ValueError("场景模型已变化，无法忠实回放该轨迹")
        directory = self.store.run_dir(run) / "replays"
        path = directory / f"{key}.npz"
        if not path.exists():
            from embodied_jev.planning import Candidate

            from .sim import execute
            poses = PoseRecorder()
            poses.capture(world)
            for record in result["records"]:
                poses.step = record["step"]
                executed, _ = execute(world, Candidate(**record["action"]), poses.capture)
                observation = world.observe()
                if (executed != record["executed"] or any(not np.allclose(observation[n], record["after"][n], atol=2e-4, rtol=0.)
                                                          for n in ("tcp", "object"))):
                    raise ValueError("历史动作重建未通过位置校验，不能作为原始 replay 展示")
            poses.save(directory, result, "verified_action_reconstruction")
        if self.renderer:
            self.renderer.close()
        self.world = world
        with np.load(path, allow_pickle=False) as archive:
            self.poses = {n: archive[n].copy() for n in archive.files}
        if self.poses["qpos"].shape[1] != world.model.nq:
            raise ValueError("Replay joint state does not match the robot")
        world.model.vis.global_.offwidth, world.model.vis.global_.offheight = 640, 400
        world.model.vis.headlight.ambient[:] = [.22, .24, .27]
        self.renderer = mujoco.Renderer(world.model, width=640, height=400)
        self.key = (run, key)

    def _metadata(self, run, key):
        self._load(run, key)
        return json.loads((self.store.run_dir(run) / "replays" / f"{key}.json").read_text())

    def metadata(self, run, key):
        return self.worker.submit(self._metadata, run, key).result(timeout=60)

    def _frame(self, run, key, frame, view):
        import mujoco
        from PIL import Image

        from .visuals import decorate
        self._load(run, key)
        if not 0 <= frame < len(self.poses["qpos"]):
            raise ValueError("Invalid replay frame")
        world = self.world
        world.data.qpos[:] = self.poses["qpos"][frame]
        world.data.time = float(self.poses["times"][frame])
        world.closed = bool(self.poses["closed"][frame])
        mujoco.mj_forward(world.model, world.data)
        camera = "wrist_camera" if view == "wrist" else mujoco.MjvCamera()
        if view != "wrist":
            camera.lookat[:] = [.25, 0., .28]
            camera.azimuth, camera.elevation, camera.distance = 135, -25, 1.75
        self.renderer.update_scene(world.data, camera=camera)
        scene = self.renderer.scene
        decorate(scene, world)
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_PLANE,
                               np.array([3., 3., .1]), np.array([0., 0., -.46]), np.eye(3).ravel(),
                               np.array([.07, .09, .115, 1.]))
            scene.ngeom += 1
        buffer = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def frame(self, run, key, frame, view="external"):
        return self.worker.submit(self._frame, run, key, frame, view).result(timeout=60)

    def close(self):
        def release():
            if self.renderer:
                self.renderer.close()
        self.worker.submit(release).result(timeout=60)
        self.worker.shutdown(wait=True)
