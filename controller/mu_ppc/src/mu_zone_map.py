"""Friction zone prior indexed by Frenet s.

Zones come from rosparam (same schema as friction_map_params used by the existing
sector_tuner stack) so we can reuse the operator's hand-labeled low/high-mu
sectors. Edges are blended with a cosine ramp to avoid gain steps at boundaries.
"""

import math


class MuZoneMap:
    def __init__(self, track_length, edge_blend=2.0, default_mu=1.0):
        self.track_length = float(track_length) if track_length else 0.0
        self.edge_blend = float(edge_blend)
        self.default_mu = float(default_mu)
        self.global_limit = 1.0
        self.zones = []  # list of (s_start, s_end, mu)

    def load_from_rosparam(self, rospy):
        try:
            n = rospy.get_param('/friction_map_params/n_sectors', 0)
            self.global_limit = float(
                rospy.get_param('/friction_map_params/global_friction_limit', 1.0))
            zones = []
            for i in range(n):
                base = f'/friction_map_params/Sector{i}'
                s0 = float(rospy.get_param(f'{base}/s_start', -1.0))
                s1 = float(rospy.get_param(f'{base}/s_end', -1.0))
                mu = float(rospy.get_param(f'{base}/friction', 1.0))
                if s0 >= 0 and s1 > s0:
                    zones.append((s0, s1, mu))
            self.zones = zones
        except Exception:
            self.zones = []

    def set_zones(self, zones):
        self.zones = [(float(a), float(b), float(c)) for (a, b, c) in zones]

    def _ramp(self, x, edge):
        if edge <= 0.0:
            return 1.0 if x >= 0.0 else 0.0
        t = max(0.0, min(1.0, x / edge))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def mu_at(self, s):
        """Return effective friction at Frenet s, blended at zone edges."""
        if not self.zones:
            return min(self.default_mu, self.global_limit)
        L = self.track_length if self.track_length > 0 else None
        if L:
            s = s % L
        mu = self.default_mu
        for (s0, s1, mu_z) in self.zones:
            inside = (s0 <= s <= s1)
            if not inside and L:
                # handle wrap-around zones
                if s0 > s1 and (s >= s0 or s <= s1):
                    inside = True
            if not inside:
                continue
            d_in = min(s - s0, s1 - s) if not (s0 > s1) else self.edge_blend
            w = self._ramp(d_in, self.edge_blend)
            mu = (1.0 - w) * self.default_mu + w * mu_z
            break
        return min(mu, self.global_limit)

    def mu_ahead(self, s, lookahead):
        """Lowest mu within [s, s+lookahead] — used as anticipatory prior."""
        if lookahead <= 0.0 or not self.zones:
            return self.mu_at(s)
        N = 8
        mu_min = self.mu_at(s)
        for k in range(1, N + 1):
            sk = s + (lookahead * k) / N
            mu_min = min(mu_min, self.mu_at(sk))
        return mu_min
