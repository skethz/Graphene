# Reference checkpoints

Download the seven original weight files from the repository's [`reference-checkpoints` release](https://github.com/skethz/Graphene/releases/tag/reference-checkpoints): `graphene-classical_x{2,3,4}.pth`, `graphene-lightweight_x{2,3,4}.pth`, and `graphene-realworld_x4.pth`.

```bash
gh release download reference-checkpoints --repo skethz/Graphene --pattern '*.pth' --dir checkpoints
python -m tools.check_checkpoints
```

Run these commands from the repository root. GitHub authentication is required while the repository is private. Files are distributed as release assets because the full-model files exceed the per-file Git size limit. The weights have not been stripped, repacked, or converted. `manifest.json` provides their byte sizes, SHA-256 hashes, and matching configurations.
