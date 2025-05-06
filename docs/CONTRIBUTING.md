# Contributing to xfvcom

Thanks for your interest in improving **xfvcom**!  
Below are the typical steps for adding code, docs, or baseline images.

---

## 1. Add examples / screenshots

1. Place images in `docs/images/`
2. Reference them from the relevant docs page (e.g. `docs/plot_2d.md`)

## 2. Update PNG baselines (only when plot appearance changes)

```bash
pytest --regenerate-baseline -q     # rebuild test images
pre-commit run --all-files          # format & type-check
git add tests/baseline/*.png
git commit -m "Update image baselines"
```

## 3. Run full test-suite

```bash
pytest -q
mypy .
```

---

[← Back to README](../README.md)

