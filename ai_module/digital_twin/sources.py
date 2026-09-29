"""Declaration of the public dataset the digital twin is built from (provenance record).

Only fixed, documented URLs are ever downloaded, and every file is verified against the
SHA-256 checksums published by PhysioNet before it is used (see ``ingest.py``)."""
from __future__ import annotations

from dataclasses import dataclass

PHYSIONET_FILES_URL = "https://physionet.org/files"


@dataclass(frozen=True)
class PackageSpec:
    """One versioned PhysioNet project folder (each has its own SHA256SUMS.txt)."""

    slug: str
    version: str
    files: tuple[str, ...]

    def url(self, relative_path: str) -> str:
        return f"{PHYSIONET_FILES_URL}/{self.slug}/{self.version}/{relative_path}"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    version: str
    license: str
    homepage: tuple[str, ...]
    citation: tuple[str, ...]
    packages: tuple[PackageSpec, ...]

    @property
    def local_dirname(self) -> str:
        return f"{self.key}-{self.version}"

    def as_provenance(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "homepage": list(self.homepage),
            "citation": list(self.citation),
            "files": [p.url(f) for p in self.packages for f in p.files],
        }


MIMIC_IV_DEMO = DatasetSpec(
    key="mimic-iv-demo",
    name="MIMIC-IV Clinical Database Demo + MIMIC-IV-ED Demo",
    version="2.2",
    license="Open Database License (ODbL) v1.0 — no credentials required",
    homepage=(
        "https://physionet.org/content/mimic-iv-demo/2.2/",
        "https://physionet.org/content/mimic-iv-ed-demo/2.2/",
    ),
    citation=(
        "Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). "
        "MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. https://doi.org/10.13026/dp1f-ex47",
        "Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). "
        "MIMIC-IV-ED Demo (version 2.2). PhysioNet. https://doi.org/10.13026/jzz5-vs76",
        "Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215-e220.",
    ),
    packages=(
        PackageSpec(
            slug="mimic-iv-demo",
            version="2.2",
            files=("LICENSE.txt", "hosp/admissions.csv.gz", "hosp/patients.csv.gz", "icu/icustays.csv.gz"),
        ),
        PackageSpec(slug="mimic-iv-ed-demo", version="2.2", files=("LICENSE.txt", "ed/edstays.csv.gz", "ed/triage.csv.gz")),
    ),
)
