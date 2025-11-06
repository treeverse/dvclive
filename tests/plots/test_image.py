import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from dvclive import Live
from dvclive.error import InvalidImageNameError
from dvclive.plots import Image as LiveImage


# From https://stackoverflow.com/questions/5165317/how-can-i-extend-image-class
class ExtendedImage(Image.Image):
    def __init__(self, img):
        self._img = img

    def __getattr__(self, key):
        return getattr(self._img, key)


def test_pil(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    live.log_image("image.png", img)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()


def test_pil_omitting_extension_doesnt_save_without_valid_format(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    with pytest.raises(
        InvalidImageNameError, match="Cannot log image with name 'whoops'"
    ):
        live.log_image("whoops", img)


def test_pil_omitting_extension_sets_the_format_if_path_given(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))

    # Save it first, we'll reload it and pass it's path to log_image again
    live.log_image("saved_with_format.png", img)

    # Now try saving without explicit format and check if the format is set correctly.
    live.log_image(
        "whoops",
        (tmp_dir / live.plots_dir / LiveImage.subfolder / "saved_with_format.png"),
    )

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "whoops.png").exists()


def test_invalid_extension(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    with pytest.raises(
        InvalidImageNameError, match="Cannot log image with name 'image\\.foo'"
    ):
        live.log_image("image.foo", img)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 3), (10, 10, 4)])
def test_numpy(tmp_dir, shape):
    from PIL import Image as ImagePIL

    live = Live()
    img = np.ones(shape, np.uint8) * 255
    live.log_image("image.png", img)

    img_path = tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    assert img_path.exists()

    val = np.asarray(ImagePIL.open(img_path))
    assert np.array_equal(val, img)


def test_path(tmp_dir):
    import numpy as np
    from PIL import Image as ImagePIL

    live = Live()
    image_data = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    pil_image = ImagePIL.fromarray(image_data)
    image_path = tmp_dir / "temp.png"
    pil_image.save(image_path)

    live = Live()
    live.log_image("foo.png", image_path)
    live.end()

    plot_file = tmp_dir / live.plots_dir / "images" / "foo.png"
    assert plot_file.exists()

    val = np.asarray(ImagePIL.open(plot_file))
    assert np.array_equal(val, image_data)


def test_override_on_step(tmp_dir):
    live = Live()

    zeros = np.zeros((2, 2, 3), np.uint8)
    live.log_image("image.png", zeros)

    live.next_step()

    ones = np.ones((2, 2, 3), np.uint8)
    live.log_image("image.png", ones)

    img_path = tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    assert np.array_equal(np.array(Image.open(img_path)), ones)


def test_cleanup(tmp_dir):
    live = Live()
    img = np.ones((10, 10, 3), np.uint8)
    live.log_image("image.png", img)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()

    Live()

    assert not (tmp_dir / live.plots_dir / LiveImage.subfolder).exists()


def test_custom_class(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    extended_img = ExtendedImage(img)
    live.log_image("image.png", extended_img)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()


def test_matplotlib(tmp_dir):
    live = Live()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    assert plt.fignum_exists(fig.number)

    live.log_image("image.png", fig)

    assert not plt.fignum_exists(fig.number)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()


@pytest.mark.parametrize("cache", [False, True])
def test_cache_images(tmp_dir, dvc_repo, cache):
    live = Live(save_dvc_exp=False, cache_images=cache)
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    live.log_image("image.png", img)
    live.end()
    assert (tmp_dir / "dvclive" / "plots" / "images.dvc").exists() == cache
