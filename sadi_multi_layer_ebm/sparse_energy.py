import torch
from multi_layer import EnergyBasedModel

class SparseEnergy(EnergyBasedModel):
    # ------------- constructor unchanged except for λ -------------------- #
    def __init__(self,
                 *args,
                 n_groups: int | None = None,
                 layers_idx: list[int] | None = None,
                 spill: str = "pad",
                 lambda_groups: float = 1e-3,   # strength of the penalty
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.lambda_groups = lambda_groups
        self.layers_idx = layers_idx if layers_idx is not None else list(range(self.n_layers))
        self.n_groups     = n_groups or min(self.model[i] for i in self.layers_idx)
        self.groups       = self._make_groups(spill)    # defined exactly as shown earlier

        # quick helpers
        self._idx_s_set  = set(self.idx_s.tolist())     # units that appear in s
        self._idx_s_map  = {j:i for i,j in enumerate(self.idx_s.tolist())}
        self._group_masks_s = [
            torch.tensor([u in g and u in self._idx_s_set for u in self.idx_s],
                         device=self.device, dtype=torch.bool)
            for g in self.groups
        ]

    # ------------- public override -------------------------------------- #
    def energy_grad(self, s, rho_u, rho_prime, y_target=None, h_target=None):
        """
        = base gradient +
          group-sparsity penalty:  λ * C_g * sign(s_i)  for every unit i in group g
          where C_g is the number of *non-zero* weights that leave/enter g.
        """
        # base gradient from the parent class
        base_grad = super().energy_grad(s, rho_u, rho_prime,
                                        y_target=y_target, h_target=h_target)


        # penalty term -----------------------------------------------------
        counts = self._cross_group_counts()              # (G,) on current weights
        penalty_grad = torch.zeros_like(s)               # [B, N_state]

        for g, mask in enumerate(self._group_masks_s):
            if not mask.any():                            # no hidden / output units in this group
                continue
            penalty_grad[:, mask] = self.lambda_groups * counts[g] * torch.sign(s[:, mask])

        return base_grad + penalty_grad


    # --------------------------------------------------------------------- #
    #                        INTERNAL HELPERS                               #
    # --------------------------------------------------------------------- #
    def _make_groups(self, spill: str) -> list[torch.Tensor]:
        """
        Partition every chosen layer’s index‑tensor into `n_groups` slices
        and fuse the k‑th slice from each layer into the k‑th group.

        Returns
        -------
        list[torch.Tensor]
            `groups[g]` holds the global node‑indices belonging to group *g*.
        """
        # Collect per‑layer chunks
        per_layer_chunks: list[list[torch.Tensor]] = []
        for l in self.layers_idx:
            layer_indices = self.idx[l]

            # torch.tensor_split → roughly equal chunks (≤1 difference)
            chunks = torch.tensor_split(layer_indices, self.n_groups)

            if spill == "trim" and len(chunks[-1]) == 0:          # rare case
                chunks = chunks[:-1]                               # drop empties
            per_layer_chunks.append(chunks)

        # Stitch chunks across layers → groups
        groups: list[torch.Tensor] = []
        for g in range(self.n_groups):
            # concatenate the g‑th chunk from *every* participating layer
            group_parts = [per_layer_chunks[l_i][g] for l_i in range(len(self.layers_idx))]
            groups.append(torch.cat(group_parts))

        return groups
    
    def _cross_group_counts(self) -> torch.Tensor:
        """
        Count (for every group) how many non-zero weights currently connect that
        group to *any* other group, restricted to the layer-to-layer matrices
        chosen in `layers_idx`.

        Returns
        -------
        torch.Tensor  shape (n_groups,)
        """
        G = len(self.groups)
        counts = torch.zeros(G, device=self.device)

        # boolean masks (global length) for quick lookup
        group_global_masks = [torch.zeros(self.total_dim, dtype=torch.bool, device=self.device)
                              .index_fill_(0, g, True) for g in self.groups]

        for l in self.layers_idx:
            if l == self.n_layers - 1:                   # no forward matrix out of final layer
                break
            W = self.W[l]                                # [size_l, size_{l+1}]
            if W.numel() == 0:
                continue

            pre_idx  = self.idx[l]
            post_idx = self.idx[l+1]

            nz_mask = (W != 0)                           # non-zero weights

            # membership matrices:  pre_in_g[:,None] & post_in_g[None,:]
            for g, gmask in enumerate(group_global_masks):
                pre_in_g  = gmask[pre_idx]
                post_in_g = gmask[post_idx]

                # connections where exactly one side is in g
                cross = nz_mask & (pre_in_g[:, None] ^ post_in_g[None, :])
                counts[g] += cross.sum()

        return counts
