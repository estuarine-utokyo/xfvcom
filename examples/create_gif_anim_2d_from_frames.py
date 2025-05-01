from xfvcom.helpers import create_gif_from_frames

frames_dir = "./frames"
output_gif = "salinity_2020_surface.gif"

create_gif_from_frames(
    frames_dir=frames_dir, output_gif=output_gif, fps=10, batch_size=200, cleanup=False
)
