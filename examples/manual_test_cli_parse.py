import sys

sys.path.insert(0, "..")

from xfvcom.cli.make_groundwater_nc import parse_forcing_value

try:
    result = parse_forcing_value("flux_by_node.csv", node_count=871)
    print(
        f"Success! Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}"
    )
    print(f"Result type: {type(result)}")
    if hasattr(result, "shape"):
        if hasattr(result, "__iter__"):
            print(
                f"Non-zero values at indices: {[i for i, v in enumerate(result) if v != 0]}"
            )
        else:
            print(f"Result value: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
