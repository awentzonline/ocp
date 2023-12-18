"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel

from . import hrr


@registry.register_model("holonet")
class HoloNet(BaseModel):
    r"""
    Holographic representation of molecular structures.

    Args:
        num_atoms (int): Number of atoms.
        bond_feat_dim (int): Dimension of bond features.
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        atom_embedding_size (int, optional): Size of atom embeddings.
            (default: :obj:`64`)
        num_graph_conv_layers (int, optional): Number of graph convolutional layers.
            (default: :obj:`6`)
        fc_feat_size (int, optional): Size of fully connected layers.
            (default: :obj:`128`)
        num_fc_layers (int, optional): Number of fully connected layers.
            (default: :obj:`4`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        num_gaussians (int, optional): Number of Gaussians used for smearing.
            (default: :obj:`50.0`)
    """

    def __init__(
        self,
        num_atoms: int,
        bond_feat_dim: int,
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        atom_embedding_size: int = 64,
        fc_feat_size: int = 128,
        num_fc_layers: int = 4,
        otf_graph: bool = False,
        cutoff: float = 6.0,
        num_elements: int = 100,
        num_axis_bins: int = 500,
        max_axis_value: float = 50.,
        # num_theta_bins: int = 500,
        # num_phi_bins: int = 250,
        # num_radius_bins: int = 500,
    ) -> None:
        super(HoloNet, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        # construct quasi-orthogonal embeddings
        self.num_elements = num_elements
        self.num_x_bins = num_axis_bins
        self.num_y_bins = num_axis_bins
        self.num_z_bins = num_axis_bins
        num_embeddings = self.num_elements + self.num_x_bins + self.num_y_bins + self.num_z_bins
        embedding_vectors = hrr.init((num_embeddings, atom_embedding_size))
        self.register_buffer(
            'element_embeddings', embedding_vectors[:self.num_elements]
        )
        offset = self.num_elements
        self.register_buffer(
            'x_embeddings', embedding_vectors[offset: offset + self.num_x_bins]
        )
        offset += self.num_x_bins
        self.register_buffer(
            'y_embeddings', embedding_vectors[offset: offset + self.num_y_bins]
        )
        offset += self.num_x_bins
        self.register_buffer(
            'z_embeddings', embedding_vectors[offset: offset + self.num_z_bins]
        )
        self.register_buffer(
            'axis_buckets',
            torch.arange(-max_axis_value, max_axis_value, 2 * max_axis_value / (num_axis_bins - 1))
        )

        # Spherical coordinates for edges
        # self.num_theta_bins = num_theta_bins
        # self.num_phi_bins = num_phi_bins
        # self.num_radius_bins = num_radius_bins
        # num_embeddings = self.num_elements + self.num_theta_bins + self.num_phi_bins + self.num_radius_bins
        # embedding_vectors = hrr.init((num_embeddings, atom_embedding_size))
        # self.register_buffer(
        #     'element_embeddings', embedding_vectors[:self.num_elements]
        # )
        # offset = self.num_elements
        # self.register_buffer(
        #     'theta_embeddings', embedding_vectors[offset: offset + self.num_theta_bins]
        # )
        # offset += self.num_theta_bins
        # self.register_buffer(
        #     'phi_embeddings', embedding_vectors[offset: offset + num_phi_bins]
        # )
        # offset += self.num_phi_bins
        # self.register_buffer(
        #     'radius_embeddings', embedding_vectors[offset: offset + num_radius_bins]
        # )

        output_layers = [
            nn.Linear(atom_embedding_size, fc_feat_size),
            nn.LeakyReLU(),
        ]
        for i in range(num_fc_layers):
            output_layers += [
                nn.Linear(fc_feat_size, fc_feat_size),
                nn.LeakyReLU(),
            ]
        output_layers += [nn.Linear(fc_feat_size, num_targets)]
        self.predict_energy = nn.Sequential(*output_layers)

    def embed_positions(self, pos):
        # bin the coordinates and look up associated embedding vector
        binned = torch.bucketize(pos, self.axis_buckets)
        binned = binned.permute(-1, 0)
        xs = self.x_embeddings[binned[0]]
        ys = self.y_embeddings[binned[1]]
        zs = self.z_embeddings[binned[2]]
        return xs + ys + zs

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        atom_embs = self.element_embeddings[data.atomic_numbers.long() - 1]
        position_embs = self.embed_positions(data.pos)
        atoms_bound = hrr.bind(atom_embs, position_embs)
        batch_feats = scatter(atoms_bound, data.batch, dim=0, reduce='sum')
        total_energy = self.predict_energy(batch_feats)
        return position_embs, total_energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        position_embs, energy = self._forward(data)

        if self.regress_forces:
            # forces = -1 * (
            #     torch.autograd.grad(
            #         energy,
            #         position_embs,
            #         grad_outputs=torch.ones_like(energy),
            #         create_graph=True,
            #     )[0]
            # )
            forces = torch.zeros_like(data.force)
            return energy, forces
        else:
            return energy