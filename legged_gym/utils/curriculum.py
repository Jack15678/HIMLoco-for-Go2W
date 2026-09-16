"""Command coverage and task-aware terrain progression (torch only)."""
import torch

TERRAINS = ('smooth_slope', 'rough_slope', 'stairs_up', 'stairs_down', 'obstacles', 'pebbles')
MODES = ('forward', 'reverse', 'left', 'right', 'yaw_left', 'yaw_right', 'mixed', 'parking')


def stratified_extended_mask(terrain_columns, fraction=.2):
    """Random, fixed membership; round to the nearest count in each terrain column."""
    if not 0 <= fraction <= 1:
        raise ValueError('extended speed fraction must be in [0, 1]')
    mask = torch.zeros_like(terrain_columns, dtype=torch.bool)
    for column in terrain_columns.unique():
        ids = (terrain_columns == column).nonzero(as_tuple=False).flatten()
        order = torch.randperm(len(ids), device=ids.device)
        mask[ids[order[:round(len(ids) * fraction)]]] = True
    return mask


class TaskCurriculum:
    """Separate terrain mastery from three per-terrain command limits.

    All scores use the command that produced the observed transition. Terrain
    promotion requires evidence for every mode on the actual target surface.
    """
    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.commands
        device, n = env.device, env.num_envs
        self.kinds = torch.as_tensor(env.terrain.column_categories, device=device)[env.terrain_types]
        self.base_limits = torch.tensor(self.cfg.base_speed_limits, device=device)
        self.max_limits = torch.tensor(self.cfg.max_speed_limits, device=device)
        self.speed_limits = self.base_limits.repeat(len(TERRAINS), 1)
        self.increments = torch.tensor(self.cfg.speed_increments, device=device)
        self.mode = torch.full((n,), 7, dtype=torch.long, device=device)
        self.age = torch.zeros(n, device=device)
        self.weights = torch.tensor(self.cfg.mode_probabilities, device=device)
        if self.weights.numel() != len(MODES) or not torch.isclose(self.weights.sum(), self.weights.new_tensor(1.)) or (self.weights < 0).any():
            raise ValueError('Command mode probabilities must be eight nonnegative values summing to one')
        if (self.base_limits <= 0).any() or (self.max_limits < self.base_limits).any():
            raise ValueError('Invalid curriculum speed bounds')
        shape = (len(TERRAINS), env.cfg.terrain.num_rows, len(MODES))
        self.terrain_seconds = torch.zeros(shape, device=device)
        self.terrain_good = torch.zeros_like(self.terrain_seconds)
        self.terrain_ready = torch.zeros(shape[:2], dtype=torch.bool, device=device)
        self.terrain_episodes = torch.zeros(shape[:2], device=device)
        self.terrain_failures = torch.zeros_like(self.terrain_episodes)
        self.edge_seconds = torch.zeros(len(TERRAINS), 3, device=device)
        self.edge_good = torch.zeros_like(self.edge_seconds)
        self.episode_seconds = torch.zeros(n, device=device)
        self.episode_good = torch.zeros(n, device=device)
        self.episode_target = torch.zeros(n, device=device)
        self.total_seconds = torch.zeros(len(TERRAINS), len(MODES), device=device)
        self.total_target_seconds = torch.zeros_like(self.total_seconds)
        self.total_squared_error = torch.zeros(len(TERRAINS), len(MODES), 3, device=device)
        self.total_good = torch.zeros_like(self.total_seconds)
        self.samples = torch.zeros(len(TERRAINS), len(MODES), 2, device=device)
        self.failures = torch.zeros_like(self.total_seconds)
        self.up_count = torch.zeros(len(TERRAINS), device=device)
        self.down_count = torch.zeros_like(self.up_count)
        self.old_new_disagreements = torch.zeros(3, 3, device=device)  # old/new down, hold, up

    def sample(self, ids):
        if not len(ids):
            return
        e = self.env
        modes = torch.multinomial(self.weights, len(ids), replacement=True)
        extended = e.extended_speed_envs[ids]
        limits = torch.where(extended[:, None], self.speed_limits[self.kinds[ids]], self.base_limits)
        low = limits.new_tensor(self.cfg.minimum_speed)
        magnitude = low + torch.rand(len(ids), 3, device=e.device) * (limits-low)
        commands = torch.zeros_like(magnitude)
        for mode, axis, sign in [(0, 0, 1), (1, 0, -1), (2, 1, 1), (3, 1, -1), (4, 2, 1), (5, 2, -1)]:
            selected = modes == mode
            commands[selected, axis] = sign * magnitude[selected, axis]
        mixed = modes == 6
        commands[mixed] = magnitude[mixed] * (2*torch.randint(0, 2, magnitude[mixed].shape, device=e.device)-1)
        e.commands[ids, :3] = commands
        e.commands[ids, 3] = 0.
        e.parking_commands[ids] = modes == 7
        self.mode[ids], self.age[ids] = modes, 0.
        index = (self.kinds[ids]*len(MODES) + modes)*2 + extended.long()
        self.samples.view(-1).index_add_(0, index, torch.ones_like(index, dtype=torch.float))

    def record(self, target_contact, inside_tile):
        e, dt = self.env, self.env.dt
        actual = torch.cat((e.base_lin_vel[:, :2], e.base_ang_vel[:, 2:3]), dim=1)
        error = (actual-e.commands[:, :3]).abs()
        tolerance = error.new_tensor(self.cfg.tracking_absolute_tolerance) + self.cfg.tracking_relative_tolerance*e.commands[:, :3].abs()
        tolerance[self.mode == 7] = error.new_tensor(self.cfg.parking_tracking_tolerance)
        good = (error <= tolerance).all(dim=1)
        self.age += dt
        steady = (self.age >= self.cfg.transition_seconds) & inside_tile
        self.episode_seconds += steady*dt
        self.episode_good += (steady & good)*dt
        self.episode_target += (steady & target_contact)*dt
        index = self.kinds*len(MODES) + self.mode
        self.total_seconds.view(-1).index_add_(0, index, torch.full_like(self.age, dt))
        self.total_good.view(-1).index_add_(0, index, good.float()*dt)
        self.total_target_seconds.view(-1).index_add_(0, index, (target_contact & inside_tile).float()*dt)
        self.total_squared_error.view(-1, 3).index_add_(0, index, error.square()*dt)
        # Normal-speed samples determine terrain difficulty; high-speed probes cannot inflate mastery.
        eligible = steady & target_contact & ~e.extended_speed_envs
        bucket = (self.kinds*e.cfg.terrain.num_rows + e.terrain_levels)*len(MODES) + self.mode
        self.terrain_seconds.view(-1).index_add_(0, bucket, eligible.float()*dt)
        self.terrain_good.view(-1).index_add_(0, bucket, (eligible & good).float()*dt)
        edge = ((e.commands[:, :3].abs() >= .8*self.speed_limits[self.kinds])
            & (steady & target_contact & e.extended_speed_envs)[:, None])
        self.edge_seconds.index_add_(0, self.kinds, edge.float()*dt)
        self.edge_good.index_add_(0, self.kinds, (edge & good[:, None]).float()*dt)
        if e.common_step_counter % round(self.cfg.curriculum_window_s/dt) == 0:
            self.advance_window()

    def advance_window(self):
        enough = (self.terrain_seconds >= self.cfg.minimum_bucket_seconds).all(dim=2)
        score = self.terrain_good / self.terrain_seconds.clamp_min(1e-9)
        failure_rate = self.terrain_failures / self.terrain_episodes.clamp_min(1)
        self.terrain_ready[:] = enough & (score >= self.cfg.promote_score).all(dim=2) & (failure_rate <= .1)
        edge_score = self.edge_good / self.edge_seconds.clamp_min(1e-9)
        enough_edge = self.edge_seconds >= self.cfg.minimum_bucket_seconds
        direction = (enough_edge & (edge_score >= self.cfg.promote_score)).float() - (enough_edge & (edge_score < self.cfg.demote_score)).float()
        self.speed_limits[:] = torch.maximum(self.base_limits, torch.minimum(self.max_limits, self.speed_limits + direction*self.increments))
        for tensor in [self.terrain_seconds, self.terrain_good, self.terrain_episodes, self.terrain_failures, self.edge_seconds, self.edge_good]:
            tensor.zero_()

    def finish(self, ids, failed):
        e = self.env
        seconds = self.episode_seconds[ids]
        score = self.episode_good[ids] / seconds.clamp_min(1e-9)
        enough = seconds >= self.cfg.minimum_episode_seconds
        up = enough & (score >= self.cfg.promote_score) & (self.episode_target[ids] >= 1.) & ~failed
        up &= self.terrain_ready[self.kinds[ids], e.terrain_levels[ids]]
        down = failed | (enough & (score < self.cfg.demote_score) & ~e.extended_speed_envs[ids])
        delta = up.long()-down.long()
        completed = ((self.age[ids] > 0) | (seconds > 0) | failed).float()
        index = self.kinds[ids]*e.cfg.terrain.num_rows + e.terrain_levels[ids]
        self.terrain_episodes.view(-1).index_add_(0, index, completed)
        self.terrain_failures.view(-1).index_add_(0, index, failed.float())
        self.failures.view(-1).index_add_(0, self.kinds[ids]*len(MODES)+self.mode[ids], failed.float())
        self.up_count.index_add_(0, self.kinds[ids], up.float())
        self.down_count.index_add_(0, self.kinds[ids], down.float())
        # Same-trajectory diagnostic only: record how the former endpoint rule would classify it.
        distance = torch.norm(e.root_states[ids, :2]-e.env_origins[ids, :2], dim=1)
        old_up = distance > e.terrain.env_length/2
        old_down = (distance < torch.norm(e.commands[ids, :2], dim=1)*e.max_episode_length_s*.5) & ~old_up
        old = old_up.long()-old_down.long()
        self.old_new_disagreements.view(-1).index_add_(0, (old+1)*3+delta+1, completed)
        self.episode_seconds[ids] = self.episode_good[ids] = self.episode_target[ids] = 0.
        self.age[ids] = 0.
        return delta

    def report(self):
        """Cumulative counters; differences between log rows reveal recent changes."""
        keys = ['speed_limits', 'samples', 'total_seconds', 'total_target_seconds', 'total_squared_error',
                'total_good', 'failures', 'up_count', 'down_count', 'old_new_disagreements']
        return dict(terrain_names=TERRAINS, mode_names=MODES,
                    **{key: getattr(self, key).detach().cpu().tolist() for key in keys})

    def state_dict(self):
        return {key: value.detach().cpu().clone() for key, value in vars(self).items() if torch.is_tensor(value)}
