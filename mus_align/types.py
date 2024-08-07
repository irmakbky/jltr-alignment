from __future__ import annotations

import abc
import enum
import functools
import hashlib
import io
import json
import pathlib
import warnings
import zipfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import librosa
import numpy as np
import pdf2image
from PIL import Image as _PILImage
from scipy.interpolate import interp1d

_PROJECT_VERSION = "0.5.0"


"""
Helpers
"""

_DISCRETIZATION = 1_000_000


def _discretize(value: float, discretization: int = _DISCRETIZATION) -> int:
    return round(value * discretization)


HashableAttributes = Union[
    Dict[str, "HashableAttributes"],
    List["HashableAttributes"],
    Tuple["HashableAttributes", ...],
    str,
    int,
    None,
]


def _sort_obj_to_hash(attrs: HashableAttributes) -> HashableAttributes:
    if attrs is None:
        raise Exception("Leaf cannot be None")
    elif isinstance(attrs, list) or isinstance(attrs, tuple):
        return [_sort_obj_to_hash(v) for v in attrs]
    elif isinstance(attrs, dict):
        return [
            (k, _sort_obj_to_hash(attrs[k]))
            for k in sorted(attrs.keys())
            if attrs[k] is not None
        ]
    elif isinstance(attrs, str) or isinstance(attrs, int):
        return attrs
    else:
        raise TypeError(f"Invalid type: {type(attrs)} {attrs}")


def _sha256(data: Union[bytes, str]) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _hash_obj(obj: HashableAttributes) -> str:
    sorted_obj = _sort_obj_to_hash(obj)
    serialized = json.dumps(sorted_obj, sort_keys=True, indent=4, default=str)
    return _sha256(serialized)


"""
Interfaces
"""


class Hashable(abc.ABC):
    @abc.abstractmethod
    def attrs_for_hash(self) -> HashableAttributes: ...

    def __hash__(self) -> int:
        return hash(self.checksum())

    def attrs_hashed(self) -> dict[str, str] | str:
        attrs_dict = self.attrs_for_hash()
        if isinstance(attrs_dict, dict):
            return {k: _hash_obj(v) for k, v in attrs_dict.items() if v is not None}
        else:
            return _hash_obj(attrs_dict)

    def checksum(self) -> str:
        return _hash_obj(self.attrs_for_hash())


class Serializable(abc.ABC):
    @abc.abstractmethod
    def as_json(self) -> dict: ...

    @classmethod
    @abc.abstractmethod
    def from_json(cls, d: dict) -> Serializable: ...


"""
Raw file types
"""


class Image(Hashable):
    def __init__(self, im: _PILImage.Image):
        self._im = im

    @property
    def pil_image(self) -> _PILImage.Image:
        return self._im

    @property
    def raw_png(self) -> bytes:
        image_bytes = io.BytesIO()
        self.pil_image.save(image_bytes, format="PNG")
        return image_bytes.getvalue()

    @property
    def width(self) -> int:
        return self.pil_image.width

    @property
    def height(self) -> int:
        return self.pil_image.height

    @property
    def size(self) -> Tuple[int, int]:
        return self.pil_image.size

    def copy(self, *args, **kwargs) -> "Image":
        return Image(im=self.pil_image.copy(*args, **kwargs))

    def crop(self, *args, **kwargs) -> "Image":
        return Image(im=self.pil_image.crop(*args, **kwargs))

    def convert(self, *args, **kwargs) -> "Image":
        return Image(im=self.pil_image.convert(*args, **kwargs))

    def putpixel(self, *args, **kwargs) -> None:
        self.pil_image.putpixel(*args, **kwargs)

    def attrs_for_hash(self) -> str:
        return _sha256(self.raw_png)

    def checksum(self) -> str:
        return self.attrs_for_hash()

    @classmethod
    def open(self, *args, **kwargs) -> "Image":
        return Image(im=_PILImage.open(*args, **kwargs))

    @classmethod
    def from_path(self, path: str | pathlib.Path) -> "Image":
        return Image(im=_PILImage.open(str(path)))


class Audio(Hashable):
    def __init__(
        self,
        *,
        samples: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        raw: Optional[bytes] = None,
        ext: Optional[str] = None,
    ):
        if samples is not None and samples.ndim != 1:
            raise ValueError()
        if samples is not None and samples.dtype != np.float32:
            raise TypeError()
        from_samples = samples is not None and sample_rate is not None
        from_raw = raw is not None
        if not (from_samples or from_raw):
            raise ValueError("Either samples and sample_rate or raw must be provided")
        self._samples = samples
        self._sample_rate = sample_rate
        self._raw = raw
        self._ext = ext

    @property
    def samples(self) -> np.ndarray:
        if self._samples is None:
            assert self._raw is not None
            self._samples, self._sample_rate = librosa.load(
                io.BytesIO(self._raw), sr=None, mono=True
            )
        return self._samples

    @property
    def sample_rate(self) -> float:
        self.samples
        assert self._sample_rate is not None
        return self._sample_rate

    @property
    def raw(self) -> bytes:
        if self._raw is None:
            raise Exception("raw is not set")
        return self._raw

    @property
    def ext(self) -> Optional[str]:
        return self._ext

    @property
    def length_seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    def attrs_for_hash(self) -> str:
        if self._raw is None:
            return _sha256(self.samples.tobytes())
        else:
            return _sha256(self.raw)

    def checksum(self) -> str:
        return self.attrs_for_hash()

    @classmethod
    def from_path(self, path: str | pathlib.Path) -> "Audio":
        with open(path, "rb") as f:
            return Audio(raw=f.read(), ext=pathlib.Path(path).suffix)


@functools.lru_cache(maxsize=8)
def _pdf_to_page_images(*args, **kwargs):
    return pdf2image.convert_from_bytes(*args, **kwargs)


class PDF(Hashable):
    def __init__(self, *, raw: bytes):
        self._raw = raw

    @property
    def raw(self) -> bytes:
        return self._raw

    def as_page_images(self, *, dpi: int = 200) -> Iterator[Image]:
        for im in _pdf_to_page_images(self.raw, dpi=dpi):
            yield Image(im=im)

    def attrs_for_hash(self) -> str:
        return _sha256(self.raw)

    def checksum(self) -> str:
        return self.attrs_for_hash()

    @classmethod
    def from_path(self, path: pathlib.Path | str) -> "PDF":
        with open(path, "rb") as f:
            return PDF(raw=f.read())


"""
Primary project types
"""


class BoundingBox(Hashable, Serializable):
    """A graphical bounding box. Coordinates are normalized to [0, 1]."""

    def __init__(
        self,
        *,
        left: float,
        top: float,
        width: float,
        height: float,
        score: Optional[float] = None,
    ):
        if left < 0 or left > 1:
            raise ValueError("left must be between 0 and 1")
        if top < 0 or top > 1:
            raise ValueError("top must be between 0 and 1")
        if width < 0 or width > 1:
            raise ValueError("width must be between 0 and 1")
        if height < 0 or height > 1:
            raise ValueError("height must be between 0 and 1")
        if left + width > 1:
            raise ValueError("right must be less than or equal to 1")
        if top + height > 1:
            raise ValueError("bottom must be less than or equal to 1")
        if score is not None and (score < 0 or score > 1):
            raise ValueError("score must be between 0 and 1")
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.score = score

    def __repr__(self) -> str:
        attrs = self.as_json()
        del attrs["score"]
        return f"BoundingBox({', '.join(f'{k}={v:.2f}' for k, v in attrs.items())})"

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def get_unique_key(self) -> str:
        o = self.attrs_for_hash()
        return f"{o['left']}-{o['top']}-{o['width']}-{o['height']}"

    def attrs_for_hash(self) -> dict:
        return {
            "left": _discretize(self.left),
            "top": _discretize(self.top),
            "width": _discretize(self.width),
            "height": _discretize(self.height),
        }

    def as_json(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "score": self.score,
        }

    @classmethod
    def from_json(cls, d) -> "BoundingBox":
        return cls(
            left=d["left"],
            top=d["top"],
            width=d["width"],
            height=d["height"],
            score=d["score"],
        )


class KeySignature(enum.Enum):
    C_MAJOR = 1
    F_MAJOR = 2
    B_FLAT_MAJOR = 3
    E_FLAT_MAJOR = 4
    A_FLAT_MAJOR = 5
    D_FLAT_MAJOR = 6
    G_FLAT_MAJOR = 7
    C_FLAT_MAJOR = 8
    G_MAJOR = 9
    D_MAJOR = 10
    A_MAJOR = 11
    E_MAJOR = 12
    B_MAJOR = 13
    F_SHARP_MAJOR = 14
    C_SHARP_MAJOR = 15

    def as_json(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, d) -> "KeySignature":
        return cls[d]


class TimeSignature:
    def __init__(self, *, numerator: int, denominator: int):
        self.numerator = numerator
        self.denominator = denominator

    def as_json(self) -> dict:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_json(cls, d) -> "TimeSignature":
        return cls(numerator=d["numerator"], denominator=d["denominator"])


class Clef(enum.Enum):
    TREBLE = 1
    BASS = 2
    ALTO = 3
    TENOR = 4

    @property
    def transposition_factor(self):
        if self == Clef.TREBLE:
            return 0
        elif self == Clef.BASS:
            return -21
        else:
            raise NotImplementedError()

    def as_json(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, d) -> "Clef":
        return cls[d]


class Measure(Hashable, Serializable):
    """A measure in a multi-page score."""

    def __init__(
        self,
        *,
        bbox: BoundingBox,
        key_signature: Optional[KeySignature] = None,
        time_signature: Optional[TimeSignature] = None,
        clefs: Optional[List[Clef]] = None,
    ):
        self.bbox = bbox
        self.key_signature = key_signature
        self.time_signature = time_signature
        self.clefs = clefs

    def __repr__(self) -> str:
        attrs: dict[str, Any] = {}
        attrs.update(self.as_json())
        attrs["bbox"] = str(self.bbox)
        return f"Measure({', '.join(f'{k}={v}' for k, v in attrs.items())})"

    def attrs_for_hash(self) -> dict:
        return {
            "bbox": self.bbox.attrs_for_hash(),
            "key_signature": (
                self.key_signature.as_json() if self.key_signature else None
            ),
            "time_signature": (
                self.time_signature.as_json() if self.time_signature else None
            ),
            "clefs": [clef.as_json() for clef in self.clefs] if self.clefs else None,
        }

    def as_json(self) -> dict:
        result: dict[str, Any] = {
            "bbox": self.bbox.as_json(),
        }
        if self.key_signature is not None:
            result["key_signature"] = self.key_signature.as_json()
        if self.time_signature is not None:
            result["time_signature"] = self.time_signature.as_json()
        if self.clefs is not None:
            result["clefs"] = [clef.as_json() for clef in self.clefs]
        return result

    @classmethod
    def from_json(cls, d) -> "Measure":
        key_signature = d.get("key_signature")
        time_signature = d.get("time_signature")
        clefs = d.get("clefs")
        return cls(
            bbox=BoundingBox.from_json(d["bbox"]),
            key_signature=(
                KeySignature.from_json(key_signature) if key_signature else None
            ),
            time_signature=(
                TimeSignature.from_json(time_signature) if time_signature else None
            ),
            clefs=[Clef.from_json(c) for c in clefs] if clefs else None,
        )


OrderedMeasure = Tuple["Page", "Measure"]
MeasureOrder = List[OrderedMeasure]


def _measure_order_attrs_for_hash(
    measure_order: MeasureOrder,
) -> HashableAttributes:
    unique_page = []
    for page, _ in measure_order:
        if page not in unique_page:
            unique_page.append(page)
    return [(unique_page.index(p), m.attrs_for_hash()) for p, m in measure_order]


class Page(Hashable, Serializable):
    """A page in a score."""

    def __init__(
        self,
        *,
        measures: List[Measure],
        image: Optional[Image] = None,
        num_staves: Optional[List[int]] = None,
    ):
        keys = [m.bbox.get_unique_key() for m in measures]
        if len(keys) != len(set(keys)):
            raise ValueError("Bounding boxes should be unique")
        self.measures = measures
        self.image = image
        if num_staves is not None:
            if len(num_staves) != self.num_lines:
                raise ValueError(
                    "num_staves must have the same length as the number of lines (systems)"
                )
        self.num_staves = num_staves

    def __len__(self) -> int:
        return len(self.measures)

    @property
    def num_lines(self) -> int:
        return len(list(self.iter_graphical_order()))

    @property
    def num_systems(self) -> int:
        return self.num_lines

    def __iter__(self) -> Iterator[Measure]:
        for line in self.iter_graphical_order():
            for measure in line:
                yield measure

    def iter_graphical_order(self) -> Iterator[List[Measure]]:
        """Iterates in *graphical* order: system-by-system ignoring repeats."""
        measures = sorted(self.measures, key=lambda m: m.bbox.top)
        while len(measures) > 0:
            line_prototype = measures[0]
            half_measure_height = line_prototype.bbox.height / 2
            line = [
                m
                for m in measures
                if abs(m.bbox.top - line_prototype.bbox.top) < half_measure_height
            ]
            line.sort(key=lambda m: m.bbox.left)
            yield line
            measures = [
                m
                for m in measures
                if abs(m.bbox.top - line_prototype.bbox.top) >= half_measure_height
            ]

    def crop_measure(self, measure: Measure) -> Image:
        if self.image is None:
            raise Exception("image must be set")
        if measure not in self.measures:
            raise ValueError("measure must be on this page")
        left = int(measure.bbox.left * self.image.width)
        top = int(measure.bbox.top * self.image.height)
        right = int(measure.bbox.right * self.image.width)
        bottom = int(measure.bbox.bottom * self.image.height)
        return self.image.crop((left, top, right, bottom))

    def lookup(self, bbox: BoundingBox) -> Optional[Measure]:
        for measure in self.measures:
            if measure.bbox.get_unique_key() == bbox.get_unique_key():
                return measure
        raise ValueError()

    def attrs_for_hash(self) -> dict:
        return {
            "measures": [m.attrs_for_hash() for m in self],
            "num_staves": self.num_staves,
            "image": self.image.attrs_for_hash() if self.image else None,
        }

    def as_json(self) -> dict:
        return {
            "measures": [m.as_json() for m in self],
            "num_staves": self.num_staves,
        }

    @classmethod
    def from_json(cls, d, *, image: Image | None = None) -> "Page":
        return cls(
            measures=[Measure.from_json(m) for m in d["measures"]],
            image=image,
            num_staves=d["num_staves"],
        )


class ScorePlayhead:
    """A score playhead representing the graphical position in a score."""

    def __init__(self, *, x: float, y: float, height: float):
        if x < 0 or x > 1:
            raise ValueError("x must be between 0 and 1")
        if y < 0 or y > 1:
            raise ValueError("y must be between 0 and 1")
        if height < 0 or height > 1:
            raise ValueError("height must be between 0 and 1")
        if y + height > 1:
            raise ValueError("y + height must be less than or equal to 1")
        self.x = x
        self.y = y
        self.height = height

    def __repr__(self) -> str:
        return (
            f"ScorePlayhead(x={self.x:.2f}, y={self.y:.2f}, height={self.height:.2f})"
        )

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def render(
        self,
        page_image: Image,
        color: str = "#FF0000",
        width: int = 3,
    ) -> Image:
        image = page_image.copy()
        for x in range(
            int(self.x * image.width - width), int(self.x * image.width + width)
        ):
            for y in range(
                int(self.y * image.height), int((self.y + self.height) * image.height)
            ):
                image.putpixel(
                    (x, y), tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                )
        return image


class Alignment(abc.ABC):
    """An alignment between a score and a performance."""

    @abc.abstractmethod
    def time_to_playhead(self, time: float) -> Tuple[Page, ScorePlayhead]:
        raise NotImplementedError()


class MeasureAwareAlignment(Alignment, Hashable):
    """An alignment between a score and a performance, with measure indices."""

    def __init__(
        self,
        *,
        measure_indices: List[float],
        times: List[float],
        logical_order: MeasureOrder,
        max_time: Optional[float] = None,
    ):
        if len(measure_indices) != len(times):
            raise ValueError("measure_indices and times must have the same length")
        if any(i > len(logical_order) for i in measure_indices):
            raise ValueError("measure_indices must be within the logical order")
        self.measure_indices = measure_indices
        self.times = times
        self._measure_to_time = interp1d(
            measure_indices, times, kind="linear", fill_value="extrapolate"
        )
        self._time_to_measure = interp1d(
            times, measure_indices, kind="linear", fill_value="extrapolate"
        )
        self.logical_order = logical_order
        self.max_time = times[-1] if max_time is None else max_time

    @property
    def length_seconds(self) -> float:
        return self.max_time

    def measure_index_to_time(self, measure_index: float) -> float:
        return self._measure_to_time(measure_index)

    def time_to_measure_index(self, time: float) -> float:
        return self._time_to_measure(time)

    def time_to_playhead(self, time: float) -> Tuple[Page, ScorePlayhead]:
        measure = max(self.time_to_measure_index(time), 0.0)
        measure_index = int(measure)
        measure_fraction = measure - measure_index
        page_obj, measure_obj = self.logical_order[measure_index]
        bbox = measure_obj.bbox
        return page_obj, ScorePlayhead(
            x=bbox.left + measure_fraction * bbox.width,
            y=bbox.top,
            height=bbox.height,
        )

    def attrs_for_hash(self) -> dict:
        return {
            "measure_indices": [_discretize(m) for m in self.measure_indices],
            "times": [_discretize(t) for t in self.times],
            "logical_order": _measure_order_attrs_for_hash(self.logical_order),
            "max_time": _discretize(self.max_time),
        }


def _decode_from_zip(
    zip: zipfile.ZipFile,
    file_name: str,
    decode_fn: Callable,
) -> Any:
    if file_name in zip.namelist():
        with zip.open(file_name) as f:
            return decode_fn(f)
    else:
        raise FileNotFoundError(f"{file_name} not found in zip file")


def _check_pages_and_logical_order_consistency(
    pages: List[Page], logical_order: MeasureOrder
):
    for page, measure in logical_order:
        if page not in pages:
            raise ValueError("Inconsistency betwen pages and logical_order")
        if sum(int(measure.bbox == page_measure.bbox) for page_measure in page) != 1:
            raise ValueError("Inconsistency betwen pages and logical_order")


class Score(Hashable):
    """A score that has been optionally labeled with a logical order."""

    def __init__(
        self,
        *,
        score_pages: List[Page],
        logical_order: Optional[MeasureOrder] = None,
        score_pdf: Optional[PDF] = None,
        transposition: Optional[int] = None,
    ):
        if logical_order is None:
            logical_order = [(page, m) for page in score_pages for m in page]
        _check_pages_and_logical_order_consistency(score_pages, logical_order)
        self.score_pages = score_pages
        self.logical_order = logical_order
        self.score_pdf = score_pdf
        self.transposition = transposition

    def __len__(self):
        return len(self.logical_order)

    def __iter__(self) -> Iterator[Tuple[Page, Measure]]:
        for p, m in self.logical_order:
            yield p, m

    def attrs_for_hash(self) -> dict:
        return {
            "score_pages": [
                {k: v for k, v in page.attrs_for_hash().items() if k != "image"}
                for page in self.score_pages
            ],
            "logical_order": _measure_order_attrs_for_hash(self.logical_order),
            "score_pdf": (self.score_pdf.attrs_for_hash() if self.score_pdf else None),
            "transposition": self.transposition,
        }

    def as_project_zip(self) -> bytes:
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, "w") as z:
            # Write version
            z.writestr("version.txt", _PROJECT_VERSION)

            # Write checksum
            z.writestr("checksum.txt", self.checksum())

            # Write pages
            z.writestr(
                "pages.json",
                json.dumps([page.as_json() for page in self.score_pages], indent=2),
            )

            # Write logical order (as coordinates)
            logical_order_as_coordinates = []
            for page, measure in self.logical_order:
                page_index = self.score_pages.index(page)
                graphical_order = list(page)
                graphical_order_index = graphical_order.index(measure)
                logical_order_as_coordinates.append((page_index, graphical_order_index))
            z.writestr("logical_order.json", json.dumps(logical_order_as_coordinates))

            # Write score PDF
            if self.score_pdf is not None:
                z.writestr("score.pdf", self.score_pdf.raw)

            # Write transposition
            if self.transposition is not None:
                z.writestr("transposition.txt", str(self.transposition))

            # Write page images
            for i, page in enumerate(self.score_pages):
                if page.image is not None:
                    z.writestr(f"frontend/page-{i}.png", page.image.raw_png)

        return zip_bytes.getvalue()

    @classmethod
    def from_project_zip(
        cls, zip_bytes: bytes, *, dpi: int = 200, check_consistency: bool = True
    ) -> "Score":
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            # Check version
            version = _decode_from_zip(z, "version.txt", lambda f: f.read().decode())
            if version != _PROJECT_VERSION:
                raise ValueError(f"Unsupported version: {version}")

            # Load score PDF
            score_pdf = None
            page_images = None
            if "score.pdf" in z.namelist():
                score_pdf = PDF(raw=z.read("score.pdf"))
                page_images = list(score_pdf.as_page_images(dpi=dpi))

            # Load pages JSON
            pages_json = _decode_from_zip(z, "pages.json", json.load)
            if page_images is not None and len(pages_json) != len(page_images):
                raise ValueError("Inconsistent number of pages in PDF and JSON")
            score_pages: List[Page] = []
            for i, d in enumerate(pages_json):
                im = page_images[i] if page_images is not None else None
                score_pages.append(Page.from_json(d, image=im))

            # Load logical order
            if "logical_order.json" in z.namelist():
                logical_order = []
                for page_index, graphical_order_index in _decode_from_zip(
                    z, "logical_order.json", json.load
                ):
                    page = score_pages[page_index]
                    measure = page.measures[graphical_order_index]
                    logical_order.append((page, measure))
            else:
                warnings.warn(
                    "Logical order not found in zip file. Defaulting to graphical order."
                )
                logical_order = None

            # Load score PDF
            score_pdf = None
            if "score.pdf" in z.namelist():
                score_pdf = PDF(raw=z.read("score.pdf"))

            # Load transposition factor
            transposition = (
                int(z.read("transposition.txt"))
                if "transposition.txt" in z.namelist()
                else None
            )

            # Load checksum
            checksum = _decode_from_zip(z, "checksum.txt", lambda f: f.read().decode())

        result = Score(
            score_pages=score_pages,
            logical_order=logical_order,
            score_pdf=score_pdf,
            transposition=transposition,
        )

        # Check consistency
        if check_consistency and checksum != result.checksum():
            raise Exception("Checksum mismatch")
        return result


class AlignedScore(Score):
    """An alignment between a score and performance audio timing."""

    def __init__(
        self,
        *,
        score_pages: List[Page],
        alignment: MeasureAwareAlignment,
        performance: Optional[Audio],
        score_pdf: Optional[PDF] = None,
    ):
        super().__init__(
            score_pages=score_pages,
            logical_order=alignment.logical_order,
            score_pdf=score_pdf,
        )
        self.alignment = alignment
        self.performance = performance

    @property
    def length_seconds(self) -> float:
        return self.alignment.length_seconds

    def attrs_for_hash(self) -> dict:
        result = super().attrs_for_hash()
        result.update(
            {
                "alignment": self.alignment.attrs_for_hash(),
                "performance": (
                    self.performance.attrs_for_hash() if self.performance else None
                ),
            }
        )
        assert result["logical_order"] == result["alignment"]["logical_order"]
        del result["logical_order"]
        return result

    def as_project_zip(self, *args, **kwargs) -> bytes:
        zip_bytes = io.BytesIO(super().as_project_zip(*args, **kwargs))
        with zipfile.ZipFile(zip_bytes, "a") as z:
            # Write audio
            if self.performance is not None:
                z.writestr(f"performance{self.performance.ext}", self.performance.raw)

            # Write alignment
            z.writestr(
                "alignment.json",
                json.dumps(
                    {
                        "measure_indices": self.alignment.measure_indices,
                        "times": self.alignment.times,
                        "max_time": self.alignment.max_time,
                    }
                ),
            )
        return zip_bytes.getvalue()

    def preview_frame(self, time: float, *args, **kwargs) -> Image:
        page, playhead = self.alignment.time_to_playhead(time)
        if page.image is None:
            raise Exception()
        return playhead.render(page.image, *args, **kwargs)

    def as_preview_frames(
        self, frame_rate: int = 30, *args, **kwargs
    ) -> Iterator[Image]:
        for t in np.arange(0, self.alignment.length_seconds, 1 / frame_rate):
            yield self.preview_frame(t)

    def as_video(self, frame_rate: int = 30, *args, **kwargs) -> bytes:
        for frame in self.as_preview_frames(frame_rate, *args, **kwargs):
            pass
        raise NotImplementedError()

    @classmethod
    def from_project_zip(
        cls, zip_bytes: bytes, *, dpi: int = 200, check_consistency: bool = True
    ) -> "AlignedScore":
        score = Score.from_project_zip(zip_bytes, dpi=dpi, check_consistency=False)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            with z.open("alignment.json") as f:
                alignment = MeasureAwareAlignment(
                    **json.load(f), logical_order=score.logical_order
                )

            performance = None
            for file_name in z.namelist():
                if file_name.startswith("performance"):
                    ext = pathlib.Path(file_name).suffix
                    performance = Audio(raw=z.read(file_name), ext=ext)

            # Load checksum
            checksum = _decode_from_zip(z, "checksum.txt", lambda f: f.read().decode())

        result = AlignedScore(
            score_pages=score.score_pages,
            alignment=alignment,
            performance=performance,
            score_pdf=score.score_pdf,
        )

        # Check consistency
        if check_consistency and checksum != result.checksum():
            raise Exception("Checksum mismatch")

        return result
