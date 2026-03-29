#!/usr/bin/env python3

"""Simple CARLA endurance test.

This script runs an ego vehicle with NPC traffic for a fixed duration and
reports whether the run completed without runtime failures.
"""

import carla
import argparse
import math
import random
import sys
import time
import weakref
from typing import List, Optional, Tuple


class SurvivalOracle:
	def __init__(self, world: carla.World, ego: carla.Vehicle, monitored_vehicle_ids: List[int]) -> None:
		self.world = world
		self.ego = ego
		self.ego_id = ego.id
		self.monitored_vehicle_ids = monitored_vehicle_ids

		self.start_time = time.monotonic()
		self._collision_events = 0
		self._reasons: List[str] = []
		self._failed = False
		self._sensor_actors: List[carla.Actor] = []

		self.setup_sensors()

	def mark_failure(self, reason: str) -> None:
		if reason not in self._reasons:
			self._reasons.append(reason)

	def setup_sensors(self) -> None:
		bp_lib = self.world.get_blueprint_library()

		collision_bp = bp_lib.find("sensor.other.collision")
		collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
		self._sensor_actors.append(collision_sensor)

		weak_self = weakref.ref(self)

		def on_collision(event: carla.CollisionEvent) -> None:
			self_ref = weak_self()
			if self_ref is None:
				return
			try:
				if self_ref._collision_events >= 2:
					self_ref._failed = True
				self_ref._collision_events += 1
				elapsed = time.monotonic() - self_ref.start_time
				self_ref.mark_failure(f"collision detected at t={elapsed:.2f}s")
			except RuntimeError as e:
				self_ref.mark_failure(f"collision callback runtime error: {e}")
		
		collision_sensor.listen(on_collision)

	@property
	def failed(self) -> bool:
		return self._failed

	@property
	def reasons(self) -> List[str]:
		return list(self._reasons)

	@property
	def collisions(self) -> int:
		return self._collision_events

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
	random.shuffle(spawn_points)

	blueprints = choose_vehicle_blueprints(world, "vehicle.*")
	npcs: List[carla.Vehicle] = []
	for sp in spawn_points:
		if len(npcs) >= count:
			break
		# Keep nearby slots free to avoid early traffic overlap.
		if sp.location.distance(ego_spawn.location) < 5.0:
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

	actor_bucket: List[carla.Actor] = []
	oracle: Optional[SurvivalOracle] = None

	try:
		spawn_points = world.get_map().get_spawn_points()

		vehicle_bps = choose_vehicle_blueprints(world, args.ego_filter)
		if not vehicle_bps:
			raise RuntimeError(f"No ego blueprints found with filter: {args.ego_filter}")

		ego_vehicle = None
		for _ in range(args.spawn_attempts):
			transform = random.choice(spawn_points)
			ego_vehicle = _spawn_vehicle(world, vehicle_bps, transform)
			if ego_vehicle is not None:
				break
		if ego_vehicle is None:
			raise RuntimeError("Failed to spawn ego vehicle")

		actor_bucket.append(ego_vehicle)
		ego_vehicle.set_autopilot(True, args.tm_port)

		vehicles = spawn_vehicles(world, args.tm_port, args.npc_count, ego_vehicle.get_transform())
		actor_bucket.extend(vehicles)
		oracle = SurvivalOracle(world=world, ego=ego_vehicle, monitored_vehicle_ids=[actor.id for actor in vehicles])

		start = time.monotonic()
		next_report = start

		while True:
			now = time.monotonic()
			elapsed = now - start

			if oracle.failed:
				print(f"[status] failure detected by oracle at t={elapsed:.1f}s", flush=True)
				return oracle

			if elapsed >= args.duration:
				print(f"[status] completed endurance window ({elapsed:.1f}s)", flush=True)
				return oracle

			if now >= next_report:
				if not ego_vehicle.is_alive:
					oracle.mark_failure("ego vehicle was destroyed")
					return oracle
				try:
					vel = ego_vehicle.get_velocity()
					speed_kmh = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
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


def main() -> int:
	argparser = argparse.ArgumentParser(description="CARLA survival test")
	argparser.add_argument("--host", default="127.0.0.1", help="CARLA host")
	argparser.add_argument("--port", type=int, default=2000, help="CARLA port")
	argparser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
	argparser.add_argument("--timeout", type=float, default=10.0, help="CARLA RPC timeout")
	argparser.add_argument("--town", default=None, help="Load map (e.g., Town05)")

	argparser.add_argument("--duration", type=float, default=120.0, help="Survival window in seconds")
	argparser.add_argument("--report-period", type=float, default=5.0, help="Progress print period")

	argparser.add_argument("--ego-filter", default="vehicle.tesla.*", help="Blueprint filter for ego vehicle")
	argparser.add_argument("--npc-count", type=int, default=20, help="Number of NPC vehicles")
	argparser.add_argument("--spawn-attempts", type=int, default=40, help="Ego spawn attempts")
	
	argparser.add_argument("--seed", type=int, default=None, help="Random seed")

	args = argparser.parse_args()
	if args.seed is not None:
		random.seed(args.seed)

	oracle = run_survival_test(args)
	if oracle is None:
		print("\n=== Survival Test Result ===")
		print("status: FAIL (runtime error during test execution)")
		return 1

	print("\n=== Survival Test Result ===")
	print(f"status: {'PASS' if not oracle.failed else 'FAIL'}")
	print(f"collisions: {oracle.collisions} collision(s) detected")
	if oracle.collisions > 0:
		print("reasons:")
		for reason in set(oracle.reasons):
			print(f" - {reason}")
	return 0 if not oracle.failed else 1


if __name__ == "__main__":
	try:
		sys.exit(main())
	except KeyboardInterrupt:
		print("\nInterrupted by user")
		sys.exit(130)
