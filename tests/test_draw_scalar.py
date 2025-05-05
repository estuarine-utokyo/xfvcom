def test_scalar_block_regression(tmp_path, fvcom_ds, plotter):
    da = fvcom_ds["temp"].isel(time=0, siglay=0)
    fig = plt.figure()
    ax = fig.add_subplot()
    triang = mtri.Triangulation(fvcom_ds["lon"], fvcom_ds["lat"], fvcom_ds["nv_zero"])
    cf = plotter._draw_scalar(ax, triang, da, opts=FvcomPlotOptions())
    assert isinstance(cf, matplotlib.contour.QuadContourSet)
