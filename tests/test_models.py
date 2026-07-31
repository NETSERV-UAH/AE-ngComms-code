import pytest
import torch

from ae_ngcomms.models import (
    Decoder,
    Encoder,
    branch_depths,
    build_autoencoder,
    hidden_widths,
    linear_macs,
)


def test_symmetric_and_asymmetric_shapes() -> None:
    inputs = torch.randn(4, 100)
    for asymmetric in (False, True):
        model = build_autoencoder(100, 25, 3, asymmetric=asymmetric)
        reconstruction, latent = model(inputs)
        assert reconstruction.shape == (4, 100)
        assert latent.shape == (4, 25)


def test_article_branch_depth_mapping() -> None:
    assert branch_depths(1, asymmetric=False) == (1, 1)
    assert branch_depths(1, asymmetric=True) == (0, 1)
    assert branch_depths(3, asymmetric=True) == (1, 2)
    assert branch_depths(5, asymmetric=True) == (2, 3)


def test_geometric_widths_and_macs_match_archived_sweep() -> None:
    assert hidden_widths(100, 25, 1) == [50]
    model = build_autoencoder(100, 25, 1, asymmetric=False)
    assert linear_macs(model.encoder) == 6250
    assert linear_macs(model.decoder) == 6250


def test_rejects_incompatible_latent_dimensions() -> None:
    encoder = Encoder(8, 2)
    decoder = Decoder(3, 8)
    with pytest.raises(ValueError, match="latent dimensions differ"):
        from ae_ngcomms.models import AsymmetricAutoencoder

        AsymmetricAutoencoder(encoder, decoder)


def test_rejects_two_normalization_layers() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Encoder(8, 2, [4], batch_norm=True, layer_norm=True)
