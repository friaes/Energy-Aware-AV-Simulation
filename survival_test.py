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
from typing import List, Optional


RESULT_JSON_PREFIX = "RESULT_JSON:"


def _is_problematic_spawn_point(
	world_map: carla.Map,
	spawn_point: carla.Transform,
	junction_clearance_m: float,
) -> bool:
	"""Return True when a spawn point is in/near a junction where conflicts are common."""
	try:
		waypoint = world_map.get_waypoint(
			spawn_point.location,
			project_to_road=True,
			lane_type=carla.LaneType.Driving,
		)
	except RuntimeError:
		return True
	if waypoint is None:
		return True
	if waypoint.is_junction:
		return True
	try:
		next_waypoints = waypoint.next(junction_clearance_m)
		if next_waypoints and any(wp.is_junction for wp in next_waypoints):
			return True
	except RuntimeError:
		return True
	return False


def _is_dense_spawn_zone(
	spawn_point: carla.Transform,
	all_spawn_points: List[carla.Transform],
	density_radius_m: float,
	max_neighbors: int,
) -> bool:
	neighbors = 0
	for candidate in all_spawn_points:
		if candidate.location.distance(spawn_point.location) > density_radius_m:
			continue
		neighbors += 1
		if neighbors > max_neighbors:
			return True
	return False


def _candidate_spawn_points(
	world: carla.World,
	spawn_points: List[carla.Transform],
	density_radius_m: float,
	max_neighbors: int,
	junction_clearance_m: float,
) -> List[carla.Transform]:
	world_map = world.get_map()
	candidates: List[carla.Transform] = []
	for sp in spawn_points:
		if _is_problematic_spawn_point(world_map, sp, junction_clearance_m):
			continue
		if _is_dense_spawn_zone(sp, spawn_points, density_radius_m, max_neighbors):
			continue
		candidates.append(sp)
	return candidates


class SurvivalOracle:
	def __init__(self, world: carla.World, ego: carla.Vehicle, args: argparse.Namespace) -> None:
		self.world = world
		self.ego = ego
		self.ego_id = ego.id
		self.front_rear_min_distance = args.front_rear_min_distance
		self.sides_min_distance = args.sides_min_distance

		self.start_time = time.monotonic()
		self._collision_events = 0
		self._lane_events = 0
		self._distance_events = 0
		self._min_observed_front_rear_distance = float("inf")
		self._min_observed_sides_distance = float("inf")
		self._last_distance_breach_time: Optional[float] = None
		self._last_distance_breach_value: Optional[float] = None
		self._reasons: List[str] = []
		self._failed = False
		self._sensor_actors: List[carla.Actor] = []

		self.setup_sensors()

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
	def min_observed_front_rear_distance(self) -> float:
		return self._min_observed_front_rear_distance

	@property
	def min_observed_sides_distance(self) -> float:
		return self._min_observed_sides_distance
	
	def mark_failure(self, reason: str) -> None:
		if reason not in self._reasons:
			self._reasons.append(reason)

	def monitor_min_distance(self, other_vehicles: List[carla.Vehicle]) -> None:
		try:
			if not self.ego.is_alive:
				return

			ego_transform = self.ego.get_transform()
			ego_location = ego_transform.location
			ego_forward = ego_transform.get_forward_vector()
			ego_right = ego_transform.get_right_vector()
			closest = float("inf")
			closest_front_rear = float("inf")
			closest_sides = float("inf")
			closest_sector = ""
			closest_threshold = 0.0

			for vehicle in other_vehicles:
				if (vehicle is None) or (not vehicle.is_alive) or (vehicle.id == self.ego_id):
					continue
				other_location = vehicle.get_transform().location
				rel = other_location - ego_location
				distance = ego_location.distance(other_location)
				longitudinal = rel.x * ego_forward.x + rel.y * ego_forward.y + rel.z * ego_forward.z
				lateral = rel.x * ego_right.x + rel.y * ego_right.y + rel.z * ego_right.z

				if abs(longitudinal) >= abs(lateral):
					if distance < closest_front_rear:
						closest_front_rear = distance
				else:
					if distance < closest_sides:
						closest_sides = distance

				if distance < closest:
					closest = distance
					if abs(longitudinal) >= abs(lateral):
						closest_sector = "front/rear"
						closest_threshold = self.front_rear_min_distance
					else:
						closest_sector = "sides"
						closest_threshold = self.sides_min_distance

			if not math.isfinite(closest):
				return

			if math.isfinite(closest_front_rear):
				self._min_observed_front_rear_distance = min(self._min_observed_front_rear_distance, closest_front_rear)
			if math.isfinite(closest_sides):
				self._min_observed_sides_distance = min(self._min_observed_sides_distance, closest_sides)

			if closest < closest_threshold:
				now = time.monotonic()
				time_gate_ok = (
					self._last_distance_breach_time is None
					or (now - self._last_distance_breach_time) >= 5.0
				)
				distance_gate_ok = (
					self._last_distance_breach_value is None
					or abs(closest - self._last_distance_breach_value) > 0.1
				)
				if time_gate_ok and distance_gate_ok:
					self._distance_events += 1
					elapsed = now - self.start_time
					self.mark_failure(
						f"minimum distance breach at t={elapsed:.2f}s "
						f"({closest_sector}: d={closest:.2f}m < {closest_threshold:.2f}m)"
					)
					self._last_distance_breach_time = now
					self._last_distance_breach_value = closest
		except RuntimeError as exc:
			self.mark_failure(f"distance monitor runtime error: {exc}")

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
				self_ref._collision_events += 1
				elapsed = time.monotonic() - self_ref.start_time
				self_ref.mark_failure(f"collision detected at t={elapsed:.2f}s")
				if self_ref._collision_events >= 5:
					self_ref._failed = True
			except RuntimeError as e:
				self_ref.mark_failure(f"collision callback runtime error: {e}")
		
		def on_lane(event: carla.LaneInvasionEvent) -> None:
			self_ref = weak_self()
			if self_ref is None:
				return
			try:
				elapsed = time.monotonic() - self_ref.start_time
				self_ref._lane_events += 1
				# if self_ref.allow_lane_invasions or elapsed < self_ref.warmup_seconds:
				# 	return
				crossing = sorted({marking.type.name for marking in event.crossed_lane_markings})
				crossing_text = ",".join(crossing) if crossing else "unknown"
				self_ref.mark_failure(f"lane invasion detected lane_mark=({crossing_text}) at t={elapsed:.2f}s")
			except RuntimeError as exc:
				self_ref.mark_failure(f"lane callback runtime error: {exc}")

		collision_sensor.listen(on_collision)
		lane_sensor.listen(on_lane)

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
	return [bp for bp in blueprints if bp.has_attribute("number_of_wheels")
			and int(bp.get_attribute("number_of_wheels")) == 4]


def _spawn_vehicle(world: carla.World, blueprints: List[carla.ActorBlueprint], transform: carla.Transform) -> Optional[carla.Vehicle]:
	blueprint = random.choice(blueprints)
	if blueprint.has_attribute("role_name"):
		blueprint.set_attribute("role_name", "ego")
	return world.try_spawn_actor(blueprint, transform)


def spawn_vehicles(world: carla.World, tm_port: int, count: int, ego_spawn: carla.Transform) -> List[carla.Vehicle]:
	if count <= 0:
		return []

	spawn_points = world.get_map().get_spawn_points()
	spawn_points = _candidate_spawn_points(
		world,
		spawn_points,
		density_radius_m=20.0,
		max_neighbors=5,
		junction_clearance_m=15.0,
	)
	random.shuffle(spawn_points)

	blueprints = choose_vehicle_blueprints(world, "vehicle.*")
	npcs: List[carla.Vehicle] = []
	for sp in spawn_points:
		if len(npcs) >= count:
			break
		# Keep nearby slots free to avoid early traffic overlap.
		if sp.location.distance(ego_spawn.location) < 20.0:
			continue
		too_close_to_existing = False
		for v in npcs:
			try:
				if (v is not None) and v.is_alive and (v.get_transform().location.distance(sp.location) < 15.0):
					too_close_to_existing = True
					break
			except RuntimeError:
				# Actor can become invalid between ticks; skip it and continue spawning.
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


def run_survival_test(args: argparse.Namespace) -> SurvivalOracle:
	client = carla.Client(args.host, args.port)
	client.set_timeout(args.timeout)

	world = client.get_world() if args.town is None else client.load_world(args.town)
	traffic_manager = client.get_trafficmanager(args.tm_port)
	if args.seed is not None:
		traffic_manager.set_random_device_seed(args.seed)
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
		spawn_points = world.get_map().get_spawn_points()
		spawn_points = _candidate_spawn_points(
			world,
			spawn_points,
			density_radius_m=20.0,
			max_neighbors=5,
			junction_clearance_m=15.0,
		)
		if not spawn_points:
			raise RuntimeError("No safe spawn points available after filtering")

		vehicle_bps = choose_vehicle_blueprints(world, args.ego_filter)
		if not vehicle_bps:
			raise RuntimeError(f"No ego blueprints found with filter: {args.ego_filter}")

		ego_vehicle = None

		start = time.monotonic()
		print(f"[start] spawning actors", flush=True)
		for _ in range(args.spawn_attempts):
			transform = random.choice(spawn_points)
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

		report2 = time.monotonic() - report - start
		print(f"[start] spawned {len(vehicles)} NPC vehicles in {report2:.2f}s", flush=True)

		oracle = SurvivalOracle(world=world, ego=ego_vehicle, args=args)

		start = time.monotonic()
		next_report = start

		while True:
			if sync_enabled:
				world.tick()
			else:
				world.wait_for_tick()
			now = time.monotonic()
			elapsed = now - start
			oracle.monitor_min_distance(vehicles)

			if oracle.failed:
				print(f"[status] failure detected by oracle at t={elapsed:3.2f}s", flush=True)
				return oracle

			if elapsed >= args.duration:
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
						print(f"[progress] t={elapsed:3.1f}s speed={speed_kmh:3.1f} km/h " f"npc={len(vehicles)}", flush=True)
				except RuntimeError as e:
					print(f"[status] runtime error while reading ego telemetry: {e}", flush=True)
					return None
				next_report = now + args.report_period

	except RuntimeError as e:
		print(f"[status] runtime failure: {e}", flush=True)
		return None

	finally:
		if oracle is not None:
			oracle.destroy()
		print('\ndestroying %d vehicles' % len(actor_bucket))
		client.apply_batch([carla.command.DestroyActor(x) for x in list(reversed(actor_bucket))])
		if sync_enabled:
			traffic_manager.set_synchronous_mode(False)
			world.apply_settings(original_settings)


def main() -> int:
	argparser = argparse.ArgumentParser(description="CARLA survival test")
	argparser.add_argument("--host", default="127.0.0.1", help="CARLA host")
	argparser.add_argument("--port", type=int, default=2000, help="CARLA port")
	argparser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
	argparser.add_argument("--timeout", type=float, default=20.0, help="CARLA RPC timeout")
	argparser.add_argument("--town", default=None, help="Load map (e.g., Town05)")

	argparser.add_argument("--duration", type=float, default=120.0, help="Survival window in seconds")
	argparser.add_argument("--report-period", type=float, default=5.0, help="Progress print period")
	argparser.add_argument("--no-progress", action="store_true", help="Disable periodic [progress] logs")
	
	argparser.add_argument("--front-rear-min-distance", type=float, default=4.0, help="Minimum allowed distance to other vehicles in front/rear sectors")
	argparser.add_argument("--sides-min-distance", type=float, default=3.0,help="Minimum allowed distance to other vehicles in left/right sectors")

	argparser.add_argument("--ego-filter", default="vehicle.tesla.*", help="Blueprint filter for ego vehicle")
	argparser.add_argument("--npc-count", type=int, default=10, help="Number of NPC vehicles")
	argparser.add_argument("--spawn-attempts", type=int, default=40, help="Ego spawn attempts")
	
	argparser.add_argument("--seed", type=int, default=None, help="Random seed")
	argparser.add_argument("--sync", dest="sync", action="store_true", help="Run world and traffic manager in synchronous mode (default: enabled)")
	argparser.add_argument("--no-sync", dest="sync", action="store_false", default=True, help="Run world in asynchronous mode")
	argparser.add_argument("--fixed-delta-seconds", type=float, default=0.05, help="Fixed simulation step when synchronous mode is enabled")

	args = argparser.parse_args()
	if args.seed is not None:
		random.seed(args.seed)

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
			"min_observed_front_rear_distance": None,
			"min_observed_side_distance": None,
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
	if math.isfinite(oracle.min_observed_front_rear_distance):
		print(f"minimum observed front/rear distance: {oracle.min_observed_front_rear_distance:.2f} m")
	else:
		print("minimum observed front/rear distance: n/a")
	if math.isfinite(oracle.min_observed_sides_distance):
		print(f"minimum observed side distance: {oracle.min_observed_sides_distance:.2f} m")
	else:
		print("minimum observed side distance: n/a")
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
		"min_observed_front_rear_distance": (
			oracle.min_observed_front_rear_distance
			if math.isfinite(oracle.min_observed_front_rear_distance)
			else None
		),
		"min_observed_side_distance": (
			oracle.min_observed_sides_distance
			if math.isfinite(oracle.min_observed_sides_distance)
			else None
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
