#!/usr/bin/env python3

"""Simple CARLA endurance test.

This script runs an ego vehicle with NPC traffic for a fixed duration and
reports whether the run completed without runtime failures.
"""

import carla
import argparse
import json
import math
import random
import sys
import time
import weakref
from pathlib import Path
from typing import List, Optional


RESULT_JSON_PREFIX = "RESULT_JSON:"

DEFAULT_CAMERA_RESOLUTION = (800, 600)
DEFAULT_CAMERA_TICK = 0.1
DEFAULT_LIDAR_TICK = 0.1
DEFAULT_LIDAR_POINTS_PER_SECOND = 10_000
DEFAULT_LIDAR_POINT_SIZE = 0.03
DEFAULT_LIDAR_POINT_LIFETIME = 0.12
DEFAULT_LIDAR_RANGE = 10.0


def resolve_weather_preset(name: str) -> Optional[carla.WeatherParameters]:
    if not name:
        return None

    normalized = name.replace("_", "").replace("-", "").casefold()
    for attr in dir(carla.WeatherParameters):
        if attr.startswith("_"):
            continue
        preset = getattr(carla.WeatherParameters, attr)
        if not isinstance(preset, carla.WeatherParameters):
            continue
        candidate = attr.replace("_", "").replace("-", "").casefold()
        if candidate == normalized:
            return preset
    return None


def parse_camera_resolution(text: str) -> tuple[int, int]:
    cleaned = text.strip().lower().replace(" ", "")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise ValueError("expected WIDTHxHEIGHT, for example 800x600")

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError("expected integer WIDTHxHEIGHT, for example 800x600") from exc

    if width <= 0 or height <= 0:
        raise ValueError("resolution values must be > 0")

    return width, height


class SurvivalOracle:
    def __init__(self, world: carla.World, ego: carla.Vehicle, args: argparse.Namespace) -> None:
        self.world = world
        self.ego = ego
        self.ego_id = ego.id
        self.args = args
        self.min_front_distance = args.min_front_distance
        self.min_distance_traveled_m = args.min_distance_traveled

        self.start_time = time.monotonic()
        self._collision_events = 0
        self._lane_events = 0
        self._distance_events = 0
        self._distance_breach_values: List[float] = []
        self._min_front_clearance = float("inf")
        self._distance_traveled_m = 0.0
        self._last_ego_location: Optional[carla.Location] = None
        self._last_distance_breach_time: Optional[float] = 0.0
        self._reasons: List[str] = []
        self._failed = False
        self._sensor_actors: List[carla.Actor] = []
        self.output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        self._camera_capture_dir = self.output_dir / "lidar_camera_captures" if self.output_dir is not None else None
        self._last_camera_image = None
        self._camera_capture_count = 0
        self._last_lidar_points = None
        self._min_lidar_clearance_m = float("inf")
        self._lidar_range = DEFAULT_LIDAR_RANGE
        self._lidar_sensor = None

        try:
            self._last_ego_location = self.ego.get_transform().location
        except RuntimeError:
            self._last_ego_location = None

        self.setup_sensors()

    def _extract_lidar_points(self, lidar_data) -> List[tuple[float, float, float]]:
        pts: List[tuple[float, float, float]] = []
        if lidar_data is None:
            return pts
        try:
            for detection in lidar_data:
                try:
                    point = detection.point
                    pts.append((float(point.x), float(point.y), float(point.z)))
                except Exception:
                    continue
        except Exception:
            return []
        return pts

    def _select_forward_clearance_points(self) -> List[tuple[float, float, float]]:
        if not self._last_lidar_points:
            return []
        try:
            lateral_thresh = 1.8  # meters half-width of the ego-lane corridor
            min_height = -2.0  # ignore road/ground returns below the vehicle body
            max_height = -1.0   # ignore high points such as overpasses, trees, etc.
            selected_points: List[tuple[float, float, float]] = []
            pts = self._extract_lidar_points(self._last_lidar_points)
            for x, y, z in pts:
                if x <= 0:
                    continue
                if abs(y) > lateral_thresh:
                    continue
                if z < min_height:
                    continue
                if z > max_height:
                    continue
                selected_points.append((x, y, z))
            return selected_points
        except Exception:
            return []

    def _compute_forward_clearance(self) -> Optional[float]:
        try:
            if self._lidar_sensor is None:
                return None
            forward_points = self._select_forward_clearance_points()
            if not forward_points:
                return float(self._lidar_range)
            return min(math.hypot(x, y) for x, y, _z in forward_points)
        except Exception:
            return None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def reasons(self) -> List[str]:
        return list(self._reasons)

    @property
    def collisions(self) -> int:
        return self._collision_events

    @property
    def lane_invasions(self) -> int:
        return self._lane_events

    @property
    def distance_breaches(self) -> int:
        return self._distance_events

    @property
    def distance_breach_values(self) -> List[float]:
        return list(self._distance_breach_values)

    @property
    def min_front_clearance(self) -> float:
        return self._min_front_clearance

    @property
    def distance_traveled_m(self) -> float:
        return self._distance_traveled_m

    def mark_failure(self, reason: str) -> None:
        if reason not in self._reasons:
            self._reasons.append(reason)

    def monitor_min_distance(self) -> None:
        try:
            if not self.ego.is_alive:
                return

            lidar_clearance = self._compute_forward_clearance()
            if lidar_clearance is None:
                return
            
            elapsed = time.monotonic() - self.start_time
            self._min_front_clearance = min(self._min_front_clearance, lidar_clearance)
            if lidar_clearance < self.min_front_distance and (elapsed - self._last_distance_breach_time) > 20.0:
                self._distance_events += 1
                self._distance_breach_values.append(lidar_clearance)
                self._failed = True
                self._last_distance_breach_time = elapsed
                self.mark_failure(
                    f"minimum LiDAR clearance breach at t={elapsed:.2f}s "
                    f"(clearance: d={lidar_clearance:.2f}m < {self.min_front_distance:.2f}m)"
                )
        except RuntimeError as exc:
            self.mark_failure(f"distance monitor runtime error: {exc}")

    def monitor_distance_traveled(self) -> None:
        try:
            if not self.ego.is_alive:
                return
            current_location = self.ego.get_transform().location
            if self._last_ego_location is not None:
                step_distance = current_location.distance(self._last_ego_location)
                if math.isfinite(step_distance) and step_distance > 0.0:
                    self._distance_traveled_m += step_distance
            self._last_ego_location = current_location
        except RuntimeError as exc:
            self.mark_failure(f"distance traveled monitor runtime error: {exc}")

    def enforce_distance_traveled_threshold(self, elapsed: float) -> None:
        if self.min_distance_traveled_m <= 0:
            return
        if self._distance_traveled_m < self.min_distance_traveled_m:
            self._failed = True
            self.mark_failure(
                f"distance traveled below threshold at t={elapsed:.2f}s "
                f"(traveled={self._distance_traveled_m:.2f}m < required={self.min_distance_traveled_m:.2f}m)"
            )

    def setup_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()

        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self._sensor_actors.append(collision_sensor)

        lane_bp = bp_lib.find("sensor.other.lane_invasion")
        lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.ego)
        self._sensor_actors.append(lane_sensor)

        weak_self = weakref.ref(self)

        def on_collision(event: carla.CollisionEvent) -> None:
            self_ref = weak_self()
            if self_ref is None:
                return
            try:
                if self_ref._collision_events < 5:
                    self_ref._collision_events += 1
                elapsed = time.monotonic() - self_ref.start_time
                self_ref.mark_failure(f"collision detected at t={elapsed:.2f}s")
                self_ref._failed = True
            except RuntimeError as exc:
                self_ref.mark_failure(f"collision callback runtime error: {exc}")

        def on_lane(event: carla.LaneInvasionEvent) -> None:
            self_ref = weak_self()
            if self_ref is None:
                return
            try:
                elapsed = time.monotonic() - self_ref.start_time
                self_ref._lane_events += 1
                crossing = sorted({marking.type.name for marking in event.crossed_lane_markings})
                crossing_text = ",".join(crossing) if crossing else "unknown"
                self_ref.mark_failure(f"lane invasion detected lane_mark=({crossing_text}) at t={elapsed:.2f}s")
            except RuntimeError as exc:
                self_ref.mark_failure(f"lane callback runtime error: {exc}")

        collision_sensor.listen(on_collision)
        lane_sensor.listen(on_lane)

        camera_enabled = self.args.camera_tick is not None or self.args.camera_resolution is not None or bool(getattr(self.args, "show_lidar_points", False))
        if camera_enabled:
            camera_width, camera_height = self.args.camera_resolution or DEFAULT_CAMERA_RESOLUTION
            camera_tick = self.args.camera_tick if self.args.camera_tick is not None else DEFAULT_CAMERA_TICK
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(camera_width))
            camera_bp.set_attribute("image_size_y", str(camera_height))
            camera_bp.set_attribute("sensor_tick", str(camera_tick))
            camera_sensor = self.world.spawn_actor(
                camera_bp,
                carla.Transform(carla.Location(x=1.5, z=2.4)),
                attach_to=self.ego,
            )
            self._sensor_actors.append(camera_sensor)

            def on_camera(image) -> None:
                self_ref = weak_self()
                if self_ref is None:
                    return
                self_ref._last_camera_image = image

            camera_sensor.listen(on_camera)

        lidar_enabled = self.args.lidar_tick is not None or self.args.lidar_points_per_second is not None
        if lidar_enabled:
            lidar_tick = self.args.lidar_tick if self.args.lidar_tick is not None else DEFAULT_LIDAR_TICK
            lidar_points_per_second = (
                self.args.lidar_points_per_second
                if self.args.lidar_points_per_second is not None
                else DEFAULT_LIDAR_POINTS_PER_SECOND
            )
            lidar_range = DEFAULT_LIDAR_RANGE
            lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("sensor_tick", str(lidar_tick))
            lidar_bp.set_attribute("range", str(lidar_range))
            self._lidar_range = lidar_range
            lidar_bp.set_attribute("points_per_second", str(lidar_points_per_second))
            lidar_sensor = self.world.spawn_actor(
                lidar_bp,
                carla.Transform(carla.Location(x=0.0, z=2.4)),
                attach_to=self.ego,
            )
            self._sensor_actors.append(lidar_sensor)
            self._lidar_sensor = lidar_sensor

            show_lidar_points = bool(getattr(self.args, "show_lidar_points", False))
            lidar_point_size = DEFAULT_LIDAR_POINT_SIZE
            lidar_point_lifetime = DEFAULT_LIDAR_POINT_LIFETIME
            lidar_debug_color = carla.Color(0, 255, 0)

            def on_lidar(points) -> None:
                self_ref = weak_self()
                if self_ref is None:
                    return
                self_ref._last_lidar_points = points
                if not show_lidar_points:
                    return
                try:
                    forward_points = self_ref._select_forward_clearance_points()
                    if not forward_points:
                        return
                    sensor_tf = lidar_sensor.get_transform()
                    drew_point = False
                    for x, y, z in forward_points:
                        world_location = sensor_tf.transform(carla.Location(x=x, y=y, z=z))
                        self_ref.world.debug.draw_point(
                            world_location,
                            size=lidar_point_size,
                            color=lidar_debug_color,
                            life_time=lidar_point_lifetime,
                            persistent_lines=False,
                        )
                        drew_point = True
                    if drew_point and self_ref._camera_capture_dir is not None and self_ref._last_camera_image is not None:
                        self_ref._camera_capture_dir.mkdir(parents=True, exist_ok=True)
                        self_ref._camera_capture_count += 1
                        capture_name = (
                            self_ref._camera_capture_dir
                            / f"lidar_capture_{self_ref._camera_capture_count:06d}_frame_{self_ref._last_camera_image.frame}.png"
                        )
                        try:
                            self_ref._last_camera_image.save_to_disk(str(capture_name))
                        except RuntimeError:
                            pass
                except RuntimeError:
                    return

            lidar_sensor.listen(on_lidar)

    def destroy(self) -> None:
        for sensor in self._sensor_actors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                sensor.destroy()
            except RuntimeError:
                pass
        self._sensor_actors.clear()


def choose_vehicle_blueprints(world: carla.World, pattern: str) -> List[carla.ActorBlueprint]:
    blueprints = world.get_blueprint_library().filter(pattern)
    return [bp for bp in blueprints if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4]


def _spawn_vehicle(world: carla.World, blueprints: List[carla.ActorBlueprint], transform: carla.Transform) -> Optional[carla.Vehicle]:
    blueprint = random.choice(blueprints)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "ego")
    return world.try_spawn_actor(blueprint, transform)


def spawn_vehicles(world: carla.World, tm_port: int, count: int, ego_spawn: carla.Transform) -> List[carla.Vehicle]:
    if count <= 0:
        return []

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    blueprints = choose_vehicle_blueprints(world, "vehicle.*")
    npcs: List[carla.Vehicle] = []
    for sp in spawn_points:
        if len(npcs) >= count:
            break
        if sp.location.distance(ego_spawn.location) < 20.0:
            continue
        too_close_to_existing = False
        for v in npcs:
            try:
                if (v is not None) and v.is_alive and (v.get_transform().location.distance(sp.location) < 15.0):
                    too_close_to_existing = True
                    break
            except RuntimeError:
                continue
        if too_close_to_existing:
            continue
        vehicle = world.try_spawn_actor(random.choice(blueprints), sp)
        if vehicle is None:
            continue
        try:
            vehicle.set_autopilot(True, tm_port)
        except RuntimeError:
            try:
                vehicle.destroy()
            except RuntimeError:
                pass
            continue
        npcs.append(vehicle)
    return npcs


def run_survival_test(args: argparse.Namespace) -> Optional[SurvivalOracle]:
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world() if args.town is None else client.load_world(args.town)
    traffic_manager = client.get_trafficmanager(args.tm_port)
    if args.seed is not None:
        traffic_manager.set_random_device_seed(args.seed)
    
    # Set weather if specified
    if args.weather is not None:
        try:
            weather = resolve_weather_preset(args.weather)
            if weather is None:
                raise ValueError(f"unknown weather preset '{args.weather}'")
            world.set_weather(weather)
            print(f"[start] weather set to {args.weather}", flush=True)
        except Exception as exc:
            print(f"[start] warning: unable to set weather '{args.weather}': {exc}", flush=True)
    
    original_settings = world.get_settings()
    sync_enabled = bool(args.sync)

    if sync_enabled:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

    actor_bucket: List[carla.Actor] = []
    oracle: Optional[SurvivalOracle] = None

    try:
        vehicle_bps = choose_vehicle_blueprints(world, args.ego_filter)
        if not vehicle_bps:
            raise RuntimeError(f"No ego blueprints found with filter: {args.ego_filter}")

        ego_vehicle = None

        start = time.monotonic()
        print("[start] spawning actors", flush=True)
        for _ in range(args.spawn_attempts):
            transform = random.choice(world.get_map().get_spawn_points())
            ego_vehicle = _spawn_vehicle(world, vehicle_bps, transform)
            if ego_vehicle is not None:
                break
        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        report = time.monotonic() - start
        print(f"[start] ego vehicle spawned in {report:.2f}s", flush=True)

        actor_bucket.append(ego_vehicle)
        ego_vehicle.set_autopilot(True, args.tm_port)

        vehicles = spawn_vehicles(world, args.tm_port, args.npc_count, ego_vehicle.get_transform())
        actor_bucket.extend(vehicles)

        report2 = time.monotonic() - start - report
        print(f"[start] spawned {len(vehicles)} NPC vehicles in {report2:.2f}s", flush=True)

        oracle = SurvivalOracle(world=world, ego=ego_vehicle, args=args)

        start = time.monotonic()
        next_report = start

        while True:
            if sync_enabled:
                world.tick()
            else:
                world.wait_for_tick()

            if not args.no_spectator:
                try:
                    spectator = world.get_spectator()
                    ego_tf = ego_vehicle.get_transform()
                    # place spectator slightly above ego so view follows clearly
                    spec_loc = ego_tf.location
                    spec_loc = carla.Location(spec_loc.x, spec_loc.y, spec_loc.z + 5.0)
                    spec_tf = carla.Transform(spec_loc, ego_tf.rotation)
                    spectator.set_transform(spec_tf)
                except Exception:
                    pass

            now = time.monotonic()
            elapsed = now - start
            oracle.monitor_min_distance()
            oracle.monitor_distance_traveled()

            if elapsed >= args.duration:
                oracle.enforce_distance_traveled_threshold(elapsed)
                print(f"[status] completed endurance window ({elapsed:.1f}s)", flush=True)
                return oracle

            if not ego_vehicle.is_alive:
                oracle.mark_failure("ego vehicle was destroyed")
                return oracle

            alive_vehicles: List[carla.Vehicle] = []
            for vehicle in vehicles:
                if vehicle is None:
                    continue
                try:
                    if not vehicle.is_alive:
                        oracle.mark_failure(f"an NPC vehicle was destroyed at t={elapsed:.2f}s")
                        if vehicle in actor_bucket:
                            actor_bucket.remove(vehicle)
                        continue
                except RuntimeError:
                    oracle.mark_failure(f"an NPC vehicle became invalid at t={elapsed:.2f}s")
                    if vehicle in actor_bucket:
                        actor_bucket.remove(vehicle)
                    continue
                alive_vehicles.append(vehicle)
            vehicles = alive_vehicles

            if now >= next_report:
                try:
                    vel = ego_vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                    if not args.no_progress:
                        print(f"[progress] t={elapsed:3.1f}s speed={speed_kmh:3.1f} km/h npc={len(vehicles)}", flush=True)
                except RuntimeError as exc:
                    print(f"[status] runtime error while reading ego telemetry: {exc}", flush=True)
                    return None
                next_report = now + args.report_period

    except Exception as exc:
        print(f"[status] runtime failure: {exc}", flush=True)
        return None

    finally:
        if oracle is not None:
            oracle.destroy()
        print('\ndestroying %d vehicles' % len(actor_bucket))
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in list(reversed(actor_bucket))])
        except Exception as exc:
            print(f"[status] cleanup warning: {exc}", flush=True)
        if sync_enabled:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass


def main() -> int:
    argparser = argparse.ArgumentParser(description="CARLA survival test")
    argparser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    argparser.add_argument("--port", type=int, default=2000, help="CARLA port")
    argparser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    argparser.add_argument("--timeout", type=float, default=20.0, help="CARLA RPC timeout")
    argparser.add_argument("--town", default=None, help="Load map (e.g., Town05)")
    argparser.add_argument("--output-dir", default="out", help="Directory where per-run artifacts such as LiDAR-triggered RGB captures will be stored")

    argparser.add_argument("--duration", type=float, default=60.0, help="Survival window in seconds")
    argparser.add_argument("--report-period", type=float, default=5.0, help="Progress print period")
    argparser.add_argument("--no-progress", action="store_true", help="Disable periodic [progress] logs")
    argparser.add_argument("--no-spectator", action="store_true", default=True, help="Do not attach the spectator view to the ego vehicle")
    argparser.add_argument("--weather", default=None, help="CARLA weather preset (e.g., HardRainSunset, ClearNoon, CloudyNoon, WetCloudyNoon, etc.)")
    
    argparser.add_argument("--camera-tick", type=float, default=None, help="Ego RGB camera sensor tick in seconds")
    argparser.add_argument("--camera-resolution", default=None, help="Ego RGB camera resolution as WIDTHxHEIGHT, for example 800x600")
    argparser.add_argument("--lidar-tick", type=float, default=None, help="Ego LiDAR sensor tick in seconds")
    argparser.add_argument("--lidar-points-per-second", type=int, default=DEFAULT_LIDAR_POINTS_PER_SECOND, help="Ego LiDAR points per second")
    argparser.add_argument("--show-lidar-points", action="store_true", default=True, help="Draw LiDAR points in the CARLA viewport in real time")
    argparser.add_argument("--min-front-distance", type=float, default=3.0, help="Minimum allowed distance to other vehicles")
    argparser.add_argument("--min-distance-traveled", type=float, default=0.0, help="Minimum required ego distance traveled in meters by the end of the run")

    argparser.add_argument("--ego-filter", default="vehicle.tesla.*", help="Blueprint filter for ego vehicle")
    argparser.add_argument("--npc-count", type=int, default=10, help="Number of NPC vehicles")
    argparser.add_argument("--spawn-attempts", type=int, default=40, help="Ego spawn attempts")

    argparser.add_argument("--seed", type=int, default=None, help="Random seed")
    argparser.add_argument("--sync", dest="sync", action="store_true", default=True, help="Run world and traffic manager in synchronous mode (default: enabled)")
    argparser.add_argument("--no-sync", dest="sync", action="store_false", help="Run world in asynchronous mode")
    argparser.add_argument("--fixed-delta-seconds", type=float, default=0.05, help="Fixed simulation step when synchronous mode is enabled")

    args = argparser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if args.camera_tick is not None and args.camera_tick <= 0:
        print("--camera-tick must be > 0", file=sys.stderr)
        return 2
    if args.lidar_tick is not None and args.lidar_tick <= 0:
        print("--lidar-tick must be > 0", file=sys.stderr)
        return 2
    if args.lidar_points_per_second is not None and args.lidar_points_per_second <= 0:
        print("--lidar-points-per-second must be > 0", file=sys.stderr)
        return 2
    if args.camera_resolution is not None:
        try:
            args.camera_resolution = parse_camera_resolution(args.camera_resolution)
        except ValueError as exc:
            print(f"--camera-resolution: {exc}", file=sys.stderr)
            return 2

    oracle = run_survival_test(args)
    if oracle is None:
        print("\n=== Survival Test Result ===")
        print("status: FAIL (runtime error during test execution)")
        payload = {
            "status": "FAIL",
            "runtime_error": True,
            "collisions": None,
            "lane_invasions": None,
            "distance_breaches": None,
            "distance_breach_values_m": None,
            "min_front_clearance": None,
            "min_front_distance": args.min_front_distance,
            "distance_traveled_m": None,
            "min_required_distance_traveled_m": None,
            "reasons": [],
        }
        print(f"{RESULT_JSON_PREFIX} {json.dumps(payload, sort_keys=True)}")
        return 1

    print("\n=== Survival Test Result ===")
    status = "PASS" if not oracle.failed else "FAIL"
    print(f"status: {status}")
    print(f"collisions: {oracle.collisions} collision(s) detected")
    print(f"lane invasions: {oracle.lane_invasions} lane invasion(s) detected")
    print(f"distance breaches: {oracle.distance_breaches} distance breach(es) detected")
    if math.isfinite(oracle.min_front_clearance):
        print(f"minimum observed front clearance: {oracle.min_front_clearance:.2f} m")
    else:
        print("minimum observed front clearance: n/a")
    print(f"distance traveled: {oracle.distance_traveled_m:.2f} m")
    if args.min_distance_traveled > 0:
        print(f"minimum required distance traveled: {args.min_distance_traveled:.2f} m")

    reasons = []
    if oracle.collisions > 0 or oracle.lane_invasions > 0 or oracle.distance_breaches > 0:
        print("reasons:")
        reasons = oracle.reasons
        for reason in reasons:
            print(f" - {reason}")

    payload = {
        "status": status,
        "runtime_error": False,
        "collisions": oracle.collisions,
        "lane_invasions": oracle.lane_invasions,
        "distance_breaches": oracle.distance_breaches,
        "distance_breach_values_m": oracle.distance_breach_values,
        "min_front_clearance": (
            oracle.min_front_clearance
            if math.isfinite(oracle.min_front_clearance)
            else None
        ),
        "min_front_distance": args.min_front_distance,
        "distance_traveled_m": oracle.distance_traveled_m,
        "min_required_distance_traveled_m": (
            args.min_distance_traveled if args.min_distance_traveled > 0 else None
        ),
        "reasons": reasons,
    }
    print(f"{RESULT_JSON_PREFIX} {json.dumps(payload, sort_keys=True)}")
    return 0 if not oracle.failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
