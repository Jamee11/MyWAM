"""Copied from the user-provided Kai0 reference; interpolation in physical EEF20."""
from collections import deque
import threading
import numpy as np

class StreamActionBuffer:
    """
    Maintains a queue of action chunks; each chunk is a deque([action0, action1, ...]).
    - New inferred chunks are appended from the right;
    - For each published step, popleft() the leftmost action from each chunk;
    - Empty chunks are dropped.
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()                 # Kept for backward compatibility
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)  # Smoothing strength (exponential weight)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()              # Current sequence to publish (after smoothing)
        self.k = 0                            # Published step count (for latency trimming)
        self.last_action = None               # Last successfully popped action

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (no longer used)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate a new inference chunk:
        1) Trim the front of the new chunk by current k and max_k (latency compensation).
        2) If there is an existing chunk (cur_chunk), apply temporal smoothing on the overlap:
           - Overlap: first element 100% old / 0% new, last element 0% old / 100% new.
           - Extra tail from the new chunk is appended.
        3) Reset k=0 as the new current execution sequence.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                # Entire chunk trimmed; skip this update
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            # Build old sequence: if empty but last_action exists, extend with last_action to min_m steps;
            # if non-empty and len < m, pad tail to min_m; if both empty, take new sequence as-is
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            # Overlap length = min of remaining old length and new length
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                # No overlap; use new sequence as-is
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            # If old sequence is longer than new, trim old tail
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # Linear weights: first element 100% old, last element 0% old
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            # Smooth the overlap region
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            # Append the extra tail from the new sequence
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    

    def pop_left_step_from_all(self):
        """Legacy interface (no longer used)."""
        with self.lock:
            if len(self.cur_chunk) > 0:
                self.cur_chunk.popleft()

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to publish; k += 1."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            # If about to pop the last element, save it as last_action
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def temporal_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        """Legacy interface (no longer used)."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            return np.asarray(self.cur_chunk[0], dtype=float)

    def delta_eef_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        # TODO: implement this
        pass
