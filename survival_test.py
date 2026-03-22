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
from typing import List, Optional


def _choose_vehicle_blueprints(world: carla.World, pattern: str) -> List[carla.ActorBlueprint]:
	blueprints = world.get_blueprint_library().filter(pattern)
	return [
		bp
		for bp in blueprints
		if bp.has_attribute("number_of_wheels")
		and int(bp.get_attribute("number_of_wheels")) == 4
	]


def _spawn_vehicle(world: carla.World, blueprints: List[carla.ActorBlueprint], transform: carla.Transform) -> Optional[carla.Vehicle]:
	blueprint = random.choice(blueprints)
	if blueprint.has_attribute("role_name"):
		blueprint.set_attribute("role_name", "hero")
	return world.try_spawn_actor(blueprint, transform)


def _spawn_vehicles(world: carla.World, tm_port: int, count: int, ego_spawn: carla.Transform) -> List[carla.Vehicle]:
	if count <= 0:
		return []

	spawn_points = world.get_map().get_spawn_points()
	random.shuffle(spawn_points)

	blueprints = _choose_vehicle_blueprints(world, "vehicle.*")
	npcs: List[carla.Vehicle] = []
	for sp in spawn_points:
		if len(npcs) >= count:
			break
		# Keep nearby slots free to avoid early traffic overlap.
		if sp.location.distance(ego_spawn.location) < 15.0:
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


def run_survival_test(args: argparse.Namespace) -> bool:
	client = carla.Client(args.host, args.port)
	client.set_timeout(args.timeout)

	world = client.get_world() if args.town is None else client.load_world(args.town)
	traffic_manager = client.get_trafficmanager(args.tm_port)

	actor_bucket: List[carla.Actor] = []

	try:
		spawn_points = world.get_map().get_spawn_points()

		vehicle_bps = _choose_vehicle_blueprints(world, args.ego_filter)
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

		vehicles = _spawn_vehicles(world, args.tm_port, args.npc_count, ego_vehicle.get_transform())
		actor_bucket.extend(vehicles)

		start = time.monotonic()
		next_report = start

		while True:
			now = time.monotonic()
			elapsed = now - start

			if elapsed >= args.duration:
				print(f"[status] completed endurance window ({elapsed:.1f}s)", flush=True)
				return True

			if now >= next_report:
				try:
					vel = ego_vehicle.get_velocity()
					speed_kmh = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
					print(f"[progress] t={elapsed:3.1f}s speed={speed_kmh:3.1f} km/h " f"npc={len(vehicles)}", flush=True)
				except RuntimeError as exc:
					print(f"[status] runtime error while reading ego telemetry: {exc}", flush=True)
					return False
				next_report = now + args.report_period

	except RuntimeError as exc:
		print(f"[status] runtime failure: {exc}", flush=True)
		return False

	finally:
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

	passed = run_survival_test(args)

	print("\n=== Survival Test Result ===")
	print(f"status: {'PASS' if passed else 'FAIL'}")
	return 0 if passed else 1


if __name__ == "__main__":
	try:
		sys.exit(main())
	except KeyboardInterrupt:
		print("\nInterrupted by user")
		sys.exit(130)
