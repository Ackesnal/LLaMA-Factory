import os
import json
import math
import torch

class RePaGateActHistogramCollector:
    """
    Collect per-layer histogram of x_gate_act = act_fn(gate_proj(x)).

    - bins: 2000
    - range: [-1, 10]
    - per-layer fixed sampling: sample_ratio (e.g., 0.1) of channels
    - store only histogram counts (int64) per layer for plotting
    """
    def __init__(
        self,
        model,
        out_dir: str,
        bins: int = 2000,
        vmin: float = -1.0,
        vmax: float = 10.0,
        sample_ratio: float = 0.1,
        seed: int = 1234,
        update_every_n_steps: int = 1,   # 每隔多少 step 更新一次统计
        save_every_n_steps: int = 200,   # 每隔多少 step 落一次盘（counts 很小，可频繁一点）
        layers_to_collect=None,          # None=全部层；或 list[int]
        device_for_hist="cpu",           # 直方图累计放 CPU，省显存
    ):
        assert 0.0 < sample_ratio <= 1.0
        assert bins > 0
        assert vmax > vmin

        self.model = model
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.bins = int(bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.sample_ratio = float(sample_ratio)
        self.seed = int(seed)
        self.update_every_n_steps = int(update_every_n_steps)
        self.save_every_n_steps = int(save_every_n_steps)
        self.layers_to_collect = set(layers_to_collect) if layers_to_collect is not None else None
        self.device_for_hist = device_for_hist

        # bin width
        self.bin_width = (self.vmax - self.vmin) / self.bins

        # state
        self.step = 0
        self.handles = []

        # per-layer fixed sampled channel indices (tensor on CPU)
        self.sampled_channels = {}  # layer_id -> LongTensor [Cs]
        # per-layer hist counts
        self.hist_counts = {}       # layer_id -> LongTensor [bins] on device_for_hist
        # per-layer total sample count (for normalization)
        self.total_values = {}      # layer_id -> int

        # build hooks
        self._register_hooks()

        # save config/meta
        self._save_meta()

    def _register_hooks(self):
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise AttributeError("Cannot find model.model.layers. Please adapt collector to your model structure.")
    
        registered = 0
        skipped_no_mlp = 0
        skipped_no_gate = 0
    
        for layer_id, layer in enumerate(layers):
            if self.layers_to_collect is not None and layer_id not in self.layers_to_collect:
                continue
    
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                skipped_no_mlp += 1
                continue
    
            gate_proj = getattr(mlp, "gate_proj", None)
            act_fn = getattr(mlp, "act_fn", None)
    
            if gate_proj is None or act_fn is None:
                skipped_no_gate += 1
                continue
    
            # self.hist_counts[layer_id] = torch.zeros(self.bins, dtype=torch.int64, device=self.device_for_hist)
            self.total_values[layer_id] = 0
    
            handle = gate_proj.register_forward_hook(self._make_gate_hook(layer_id))
            self.handles.append(handle)
            registered += 1
    
        print(
            f"[RePaGateActHistogramCollector] layers={len(layers)}, "
            f"hooks_registered={registered}, skipped_no_mlp={skipped_no_mlp}, skipped_no_gate_or_act={skipped_no_gate}",
            flush=True,
        )

    def _make_gate_hook(self, layer_id: int):
        def hook(module, inp, out):
            # out: gate_proj(x) with shape [B, N, C]
            if (self.step % self.update_every_n_steps) != 0:
                return
    
            with torch.no_grad():
                layers = self.model.model.layers
                mlp = layers[layer_id].mlp
    
                # IMPORTANT: use `out` to avoid recursion
                x_gate_act = mlp.act_fn(out)  # [B, N, C]
                C = x_gate_act.shape[-1]
    
                # fixed per-layer channel sampling (only once)
                if layer_id not in self.sampled_channels:
                    Cs = max(1, int(math.floor(C * self.sample_ratio)))
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.seed + layer_id * 1000003)
                    perm = torch.randperm(C, generator=g)[:Cs]
                    self.sampled_channels[layer_id] = perm.to(device="cpu", dtype=torch.int64)
    
                ch_cpu = self.sampled_channels[layer_id]            # [Cs] on CPU
                Cs = int(ch_cpu.numel())
                ch = ch_cpu.to(device=x_gate_act.device)            # [Cs] on device
    
                # select sampled channels: [B, N, Cs]
                x_sel = x_gate_act.index_select(dim=2, index=ch)
    
                # finite mask
                finite = torch.isfinite(x_sel)
                if not finite.any():
                    return
    
                # clamp -> bin index in [0, bins-1], invalid set to -1
                x_sel = torch.clamp(x_sel, min=self.vmin, max=self.vmax)
                idx = torch.floor((x_sel - self.vmin) / self.bin_width).to(torch.int64)
                idx = torch.clamp(idx, 0, self.bins - 1)
                idx = torch.where(finite, idx, torch.full_like(idx, -1))  # invalid -> -1
    
                # reshape to [T, Cs], where T = B*N
                idx_tc = idx.view(-1, Cs)
    
                # init per-channel histogram storage: [Cs, bins] on CPU
                # (if you previously had 1D hist_counts, this overwrites to 2D)
                if (layer_id not in self.hist_counts) or (self.hist_counts[layer_id].ndim != 2) or (self.hist_counts[layer_id].shape[0] != Cs):
                    self.hist_counts[layer_id] = torch.zeros((Cs, self.bins), dtype=torch.int64, device="cpu")
    
                # cache channel offsets on CPU: offset[c] = c * bins
                if not hasattr(self, "_ch_offsets_cpu"):
                    self._ch_offsets_cpu = {}
                if layer_id not in self._ch_offsets_cpu or self._ch_offsets_cpu[layer_id].numel() != Cs:
                    self._ch_offsets_cpu[layer_id] = (torch.arange(Cs, dtype=torch.int64) * self.bins).view(1, Cs)
    
                offsets = self._ch_offsets_cpu[layer_id]  # [1, Cs] on CPU
    
                # optional chunking to reduce peak CPU memory
                chunk_T = getattr(self, "hist_chunk_tokens", 65536)  # you can set self.hist_chunk_tokens outside
                T = idx_tc.shape[0]
    
                for start in range(0, T, chunk_T):
                    end = min(start + chunk_T, T)
                    chunk = idx_tc[start:end]              # [t, Cs] on device
                    chunk_cpu = chunk.to(device="cpu")     # [t, Cs] on CPU
    
                    valid = (chunk_cpu >= 0)
                    if not valid.any():
                        continue
    
                    # keys = c*bins + bin_id, then one bincount gives [Cs*bins]
                    keys = (chunk_cpu + offsets)           # [t, Cs]
                    keys = keys[valid]                     # [num_valid]
                    bc = torch.bincount(keys, minlength=Cs * self.bins).to(torch.int64)
                    self.hist_counts[layer_id] += bc.view(Cs, self.bins)
    
                # total number of valid values (across all sampled channels)
                self.total_values[layer_id] += int(finite.sum().item())
    
        return hook


    def step_end(self):
        """Call once per training step (or forward iteration)."""
        self.step += 1
        if (self.step % self.save_every_n_steps) == 0:
            self.save()

    def save(self):
        """Save per-layer histograms + sampled channels."""
        payload = {
            "bins": self.bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "bin_width": self.bin_width,
            "step": self.step,
            "layers": {},
        }
        for layer_id, hist in self.hist_counts.items():
            payload["layers"][int(layer_id)] = {
                "hist_counts": hist.cpu(),
                "total_values": int(self.total_values[layer_id]),
                "sampled_channels": self.sampled_channels[layer_id].cpu() if layer_id in self.sampled_channels else None,
            }

        path = os.path.join(self.out_dir, f"x_gate_act_hist_step{self.step:08d}.pt")
        torch.save(payload, path)

    def close(self):
        """Flush + remove hooks."""
        self.save()
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _save_meta(self):
        meta = {
            "bins": self.bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "sample_ratio": self.sample_ratio,
            "seed": self.seed,
            "update_every_n_steps": self.update_every_n_steps,
            "save_every_n_steps": self.save_every_n_steps,
            "layers_to_collect": sorted(list(self.layers_to_collect)) if self.layers_to_collect is not None else None,
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

