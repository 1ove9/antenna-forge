#!/usr/bin/env python3
"""
Seed the YAF material library with standard antenna materials.

Run: python scripts/seed_materials.py
"""

from __future__ import annotations

from yaf_core.physics.materials import MaterialLibrary


def main() -> None:
    lib = MaterialLibrary()
    materials = lib.list_all()
    print(f"Seeded {len(materials)} materials:")
    for mid in materials:
        mat = lib.get(mid)
        print(f"  {mid:20s}  {mat.name:35s}  εr={mat.epsilon_r:.2f}  σ={mat.sigma:.2e}")

    # Test Kubo formula
    sigma = lib.get_dispersive_permittivity("graphene", 1e12)
    print(f"\nGraphene σ(1 THz) = {sigma:.6e}")

    # Test Drude model
    eps = lib.get_dispersive_permittivity("plasma_ar", 1e10)
    print(f"Plasma ε(10 GHz) = {eps:.6f}")


if __name__ == "__main__":
    main()
